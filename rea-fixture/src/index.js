import { normalizeQuery, searchItems } from "./search.js";

export function createSearch(items) {
  return (query) => searchItems(items, normalizeQuery(query));
}
