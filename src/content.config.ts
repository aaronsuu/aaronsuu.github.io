import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const projects = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    thumbnail: z.string().optional(),
    heroImage: z.string().optional(),
    coverHtml: z.string().optional(),
    flag: z.enum(['claude-assisted', 'claude-researched', 'independent']).default('independent'),
    keywords: z.array(z.string()),
    minorTags: z.array(z.string()).optional(),
    images: z.array(z.string()).optional(),
    order: z.number().optional(),
    draft: z.boolean().optional(),
  }),
});

export const collections = { projects };
