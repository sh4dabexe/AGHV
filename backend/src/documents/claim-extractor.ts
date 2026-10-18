import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import { GoogleGenAI } from '@google/genai';
import { DocClaim } from '../typesafe/comparator';
import { DocumentStore } from './store';
import { Claim } from '../shared/types';
import { EvidenceEngine } from '../evidence/engine';

/**
 * Remove markdown code fences and whitespace from model JSON output.
 */
export function cleanJsonText(text: string): string {
  let cleaned = text.trim();
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '');
  }
  return cleaned.trim();
}

const EXTRACT_PROMPT = `You are a software documentation analyst.
Your task is to extract every distinct, verifiable SOFTWARE CLAIM from the provided document.

A software claim is a statement that asserts that the software:
- Has a specific feature or capability
- Implements a specific technology or pattern
- Satisfies a performance or security property
- Integrates with a specific third-party service

Rules:
- Extract ONLY factual claims about the software, NOT marketing language
- Each claim must be a complete, self-contained sentence
- Ignore vague statements like "easy to use" or "powerful platform"
- Return ONLY valid JSON in this exact format, with no extra text:

{
  "claims": [
    { "text": "claim text", "section": "section heading or empty string" }
  ]
}

Document:
`;

/**
 * Use Gemini to extract verifiable software claims from document text.
 */
export async function extractClaimsFromText(
  docId: string,
  docText: string
): Promise<DocClaim[]> {
  const apiKey = process.env.GEMINI_API_KEY || process.env.gemeni_api || process.env.GEMINI_API;
  if (!apiKey) {
    console.warn('[ClaimExtractor] GEMINI_API_KEY not set — returning empty claims');
    return [];
  }

  // Truncate to ~60 000 chars to stay within context window
  const truncated = docText.slice(0, 60_000);
  const ai = new GoogleGenAI({ apiKey });

  let response: any;
  try {
    response = await ai.models.generateContent({
      model: process.env.GEMINI_MODEL || 'gemini-3.5-flash',
      contents: EXTRACT_PROMPT + truncated,
      config: { responseMimeType: 'application/json' }
    });
  } catch (err: any) {
    console.error('[ClaimExtractor] Gemini call failed:', err?.message);
    return [];
  }

  if (!response?.text) {
    console.warn('[ClaimExtractor] Gemini returned empty text');
    return [];
  }

  try {
    const cleaned = cleanJsonText(response.text);
    const parsed = JSON.parse(cleaned) as {
      claims: Array<{ text: string; section?: string }>;
    };

    return (parsed.claims || []).map((c, idx) => ({
      claim_id: `CLAIM-${crypto.createHash('sha256')
        .update(`${docId}-${idx}-${c.text}`)
        .digest('hex')
        .substring(0, 8)
        .toUpperCase()}`,
      doc_id: docId,
      text: c.text,
      section: c.section ?? '',
      line_number: undefined,
    }));
  } catch (err) {
    console.error('[ClaimExtractor] Failed to parse Gemini JSON response:', err);
    return [];
  }
}

export interface ClaimExtractionOptions {
  repoPath: string;
  projectId?: string;
  projectMap?: any;
  engine?: EvidenceEngine;
}

const REPO_DOCS_PROMPT = `You are an expert software specification and architecture analyst.
Your task is to analyze the provided documentation, specifications, and package manifest from a software repository, and extract EVERY distinct, verifiable SOFTWARE CLAIM.

A verifiable software claim is a specific assertion about what the software implements or provides:
1. Feature/Capability claims (e.g., "Supports user authentication via Google OAuth and Supabase", "Generates interactive Mermaid diagrams", "Exports reports as PDF and Markdown")
2. API & Endpoint claims (e.g., "Provides REST API endpoints for repository analysis under /api/v1", "Validates requests with Zod schemas")
3. Architecture & Tech Stack claims (e.g., "Built with Express.js and TypeScript on Node.js", "Frontend uses React and Tailwind CSS")
4. Database & Persistence claims (e.g., "Persists analysis reports in Supabase PostgreSQL database", "Caches repository ingest dumps with TTL")
5. Security claims (e.g., "Redacts API keys and secrets before logging or LLM transmission", "Enforces CORS and rate limits")
6. Testing & Quality claims (e.g., "Includes unit and integration test suite using Jest/Vitest", "Provides Docker sandbox execution for safe test runs")
7. Deployment & Tooling claims (e.g., "Configured for deployment on Render and Vercel", "Uses Docker container for runtime isolation")

Rules:
- Extract factual claims that can be checked against the repository code, tests, or configurations.
- NEVER invent claims or copy unrelated template text.
- Do NOT extract marketing fluff like "super fast", "world class", or "revolutionary".
- Make each claim statement clear, concise, and complete.
- Categorize each claim into one of: 'feature', 'api', 'security', 'architecture', 'testing', 'deployment', 'data_handling'.
- Set importance to one of: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'.
- Identify the source file (e.g. "README.md", "specification.md", "package.json") and section where available.

Return ONLY valid JSON matching this schema:
{
  "claims": [
    {
      "text": "Specific, verifiable claim statement",
      "category": "feature|api|security|architecture|testing|deployment|data_handling",
      "importance": "CRITICAL|HIGH|MEDIUM|LOW",
      "sourceType": "documentation|README|source_code|package_manifest",
      "sourceFile": "filename",
      "sourceLocation": "section heading or empty string",
      "reasoning": "What code or configuration evidence verifies this claim"
    }
  ]
}

Repository Context & Documentation:
`;

export class ClaimExtractorService {
  /**
   * Automatically discover and extract claims from uploaded documents and repository documentation.
   */
  public static async extractFromRepositoryAndDocs(options: ClaimExtractionOptions): Promise<Claim[]> {
    const { repoPath, projectId, projectMap, engine } = options;
    const documentsContent: Array<{ source: string; filename: string; text: string }> = [];

    // 1. Gather uploaded documents from DocumentStore
    if (projectId) {
      const uploadedDocs = DocumentStore.getAll(projectId);
      for (const doc of uploadedDocs) {
        if (doc.content && doc.content.trim().length > 0) {
          documentsContent.push({
            source: 'uploaded_doc',
            filename: doc.filename,
            text: doc.content
          });
          if (engine) {
            engine.addEvidence(
              'DOCUMENT',
              doc.filename,
              doc.content.substring(0, 10000),
              `Uploaded document: ${doc.filename}`,
              'user_upload',
              undefined,
              undefined,
              { doc_id: doc.doc_id, size_bytes: doc.size_bytes }
            );
          }
        }
      }
    }

    // 2. Discover documentation files in the repository
    const candidateFiles = [
      'README.md',
      'README',
      'readme.md',
      'readme.markdown',
      'README.txt',
      'ARCHITECTURE.md',
      'SPEC.md',
      'SPECIFICATION.md',
      'CONTRIBUTING.md',
      'DESIGN.md'
    ];

    for (const cand of candidateFiles) {
      try {
        const filePath = path.join(repoPath, cand);
        const stat = await fs.stat(filePath).catch(() => null);
        if (stat && stat.isFile()) {
          const content = await fs.readFile(filePath, 'utf-8');
          if (content.trim().length > 0) {
            documentsContent.push({
              source: 'repository_doc',
              filename: cand,
              text: content
            });
            if (engine) {
              engine.addEvidence(
                'DOCUMENT',
                cand,
                content.substring(0, 10000),
                `Repository documentation file: ${cand}`,
                'repository'
              );
            }
          }
        }
      } catch (err) {
        // file doesn't exist, continue
      }
    }

    // Also check docs/ or documentation/ subdirectories
    for (const docDir of ['docs', 'doc', 'documentation', '.github']) {
      try {
        const dirPath = path.join(repoPath, docDir);
        const entries = await fs.readdir(dirPath, { withFileTypes: true }).catch(() => []);
        for (const entry of entries) {
          if (entry.isFile() && (entry.name.endsWith('.md') || entry.name.endsWith('.txt'))) {
            try {
              const fullPath = path.join(dirPath, entry.name);
              const relPath = path.join(docDir, entry.name).replace(/\\/g, '/');
              const content = await fs.readFile(fullPath, 'utf-8');
              if (content.trim().length > 0) {
                documentsContent.push({
                  source: 'repository_doc',
                  filename: relPath,
                  text: content
                });
                if (engine) {
                  engine.addEvidence(
                    'DOCUMENT',
                    relPath,
                    content.substring(0, 8000),
                    `Documentation file: ${relPath}`,
                    'repository'
                  );
                }
              }
            } catch {
              // ignore
            }
          }
        }
      } catch {
        // ignore
      }
    }

    // 3. Include package.json manifest context if present
    let packageInfo: any = null;
    try {
      const pkgPath = path.join(repoPath, 'package.json');
      const pkgStr = await fs.readFile(pkgPath, 'utf-8');
      packageInfo = JSON.parse(pkgStr);
    } catch {
      // no package.json or unparseable
    }

    // 4. Build prompt context
    let promptContext = '';
    for (const doc of documentsContent) {
      promptContext += `\n--- SOURCE: ${doc.filename} (${doc.source}) ---\n`;
      promptContext += doc.text.slice(0, 15000); // limit per file
      promptContext += '\n';
    }

    if (packageInfo) {
      promptContext += `\n--- SOURCE: package.json (package manifest) ---\n`;
      promptContext += JSON.stringify({
        name: packageInfo.name,
        description: packageInfo.description,
        scripts: packageInfo.scripts,
        dependencies: Object.keys(packageInfo.dependencies || {}),
        devDependencies: Object.keys(packageInfo.devDependencies || {})
      }, null, 2);
      promptContext += '\n';
    }

    if (projectMap) {
      promptContext += `\n--- SOURCE: Project Structure Mapping ---\n`;
      promptContext += JSON.stringify({
        languages: projectMap.languages,
        frameworks: projectMap.frameworks,
        packageManagers: projectMap.packageManagers,
        entryPoints: projectMap.entryPoints,
        directories: projectMap.directories
      }, null, 2);
      promptContext += '\n';
    }

    const apiKey = process.env.GEMINI_API_KEY || process.env.gemeni_api || process.env.GEMINI_API;
    const extractedClaims: Claim[] = [];

    // 5. Query Gemini for Claim Extraction if context exists and API key is present
    if (apiKey && promptContext.trim().length > 50) {
      try {
        const ai = new GoogleGenAI({ apiKey });
        const modelName = process.env.GEMINI_MODEL || 'gemini-3.5-flash';
        const response = await ai.models.generateContent({
          model: modelName,
          contents: REPO_DOCS_PROMPT + promptContext.slice(0, 60000),
          config: { responseMimeType: 'application/json' }
        });

        if (response?.text) {
          const cleaned = cleanJsonText(response.text);
          const parsed = JSON.parse(cleaned);
          const rawClaims = Array.isArray(parsed.claims) ? parsed.claims : (Array.isArray(parsed) ? parsed : []);

          for (const item of rawClaims) {
            if (item && typeof item.text === 'string' && item.text.trim().length > 5) {
              const text = item.text.trim();
              const hashId = crypto.createHash('sha256').update(text.toLowerCase()).digest('hex').substring(0, 8).toUpperCase();
              extractedClaims.push({
                claim_id: `CLAIM-${hashId}`,
                text,
                category: item.category || 'feature',
                importance: (['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].includes(item.importance) ? item.importance : 'MEDIUM') as any,
                sourceType: item.sourceType || 'documentation',
                sourceFile: item.sourceFile || 'README.md',
                sourceLocation: item.sourceLocation || '',
                reasoning: item.reasoning || 'Verify implementation against source AST and execution evidence.',
                recommendedAction: item.recommendedAction
              });
            }
          }
        }
      } catch (err: any) {
        console.warn('[ClaimExtractorService] Gemini claim extraction call failed:', err?.message || err);
      }
    }

    // 6. Fallback / Augment: If no claims extracted or Gemini unavailable, generate deterministic claims from project architecture
    if (extractedClaims.length === 0) {
      console.log('[ClaimExtractorService] Generating deterministic architectural claims from repository mapping...');
      
      if (projectMap?.frameworks?.length > 0) {
        for (const fw of projectMap.frameworks) {
          const text = `Application architecture is built on ${fw} framework.`;
          const hashId = crypto.createHash('sha256').update(text.toLowerCase()).digest('hex').substring(0, 8).toUpperCase();
          extractedClaims.push({
            claim_id: `CLAIM-${hashId}`,
            text,
            category: 'architecture',
            importance: 'HIGH',
            sourceType: 'source_code',
            sourceFile: 'package.json',
            reasoning: `Inspect dependencies and entry points for ${fw} initialization.`
          });
        }
      }

      if (projectMap?.entryPoints?.length > 0) {
        const text = `Defines runnable application entry point at ${projectMap.entryPoints[0]}.`;
        const hashId = crypto.createHash('sha256').update(text.toLowerCase()).digest('hex').substring(0, 8).toUpperCase();
        extractedClaims.push({
          claim_id: `CLAIM-${hashId}`,
          text,
          category: 'architecture',
          importance: 'CRITICAL',
          sourceType: 'source_code',
          sourceFile: projectMap.entryPoints[0],
          reasoning: 'Verify that the entry point file exists and initializes application lifecycle.'
        });
      }

      if (packageInfo?.scripts?.test) {
        const text = `Automated testing suite is configured via npm test script: "${packageInfo.scripts.test}".`;
        const hashId = crypto.createHash('sha256').update(text.toLowerCase()).digest('hex').substring(0, 8).toUpperCase();
        extractedClaims.push({
          claim_id: `CLAIM-${hashId}`,
          text,
          category: 'testing',
          importance: 'HIGH',
          sourceType: 'package_manifest',
          sourceFile: 'package.json',
          reasoning: 'Verify test configuration, test files, and test execution results.'
        });
      }

      if (packageInfo?.description) {
        const text = `Project provides: ${packageInfo.description}`;
        const hashId = crypto.createHash('sha256').update(text.toLowerCase()).digest('hex').substring(0, 8).toUpperCase();
        extractedClaims.push({
          claim_id: `CLAIM-${hashId}`,
          text,
          category: 'feature',
          importance: 'MEDIUM',
          sourceType: 'package_manifest',
          sourceFile: 'package.json',
          reasoning: 'Inspect implementation of stated project functionality.'
        });
      }

      // If still empty (e.g. bare repo with no package.json), add a baseline codebase claim
      if (extractedClaims.length === 0 && repoPath) {
        const text = 'Repository contains readable source files and directory structure.';
        const hashId = crypto.createHash('sha256').update(text.toLowerCase()).digest('hex').substring(0, 8).toUpperCase();
        extractedClaims.push({
          claim_id: `CLAIM-${hashId}`,
          text,
          category: 'architecture',
          importance: 'LOW',
          sourceType: 'source_code',
          sourceFile: 'root',
          reasoning: 'Verify repository contents and directory layout.'
        });
      }
    }

    // 7. Deduplicate claims
    const seen = new Set<string>();
    const deduplicated: Claim[] = [];
    for (const c of extractedClaims) {
      const norm = c.text.toLowerCase().replace(/[^a-z0-9]/g, ' ').replace(/\s+/g, ' ').trim();
      if (!seen.has(norm)) {
        seen.add(norm);
        deduplicated.push(c);
      }
    }

    return deduplicated;
  }
}
