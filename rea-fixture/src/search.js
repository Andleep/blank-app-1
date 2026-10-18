export function normalizeQuery(value) {
  return String(value ?? "").trim().toLowerCase().replace(/\\s+/g, " ");
}

export function searchItems(items, query) {
  const needle = normalizeQuery(query);
  if (!needle) return items;
  return items.filter((item) => normalizeQuery(item.name).includes(needle));
}
