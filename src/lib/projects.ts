import { getCollection } from 'astro:content';
// Rankings live in src/data/projects.csv — edit it to reorder or promote projects.
// Imported as a raw string so the contents are inlined at build time; reading from
// disk at runtime breaks in production, where this module is bundled into dist/.
import csvRaw from '../data/projects.csv?raw';

type RankEntry = { rank: string; lastUpdated: string };

const rankMap: Record<string, RankEntry> = {};
for (const line of csvRaw.trim().split('\n').slice(1)) {
  const [id, rank, lastUpdated] = line.split(',').map((s) => s.trim());
  if (id) rankMap[id] = { rank, lastUpdated };
}

export const getRank = (id: string) => rankMap[id]?.rank ?? '';

const rankNum = (id: string) => {
  const n = parseInt(getRank(id).replace('P', ''));
  return isNaN(n) ? 999 : n;
};

/** Published projects, sorted by CSV rank, then by frontmatter `order`. */
export async function getRankedProjects() {
  const projects = await getCollection('projects', ({ data }) => !data.draft);
  return projects.sort(
    (a, b) => rankNum(a.id) - rankNum(b.id) || (a.data.order ?? 999) - (b.data.order ?? 999),
  );
}
