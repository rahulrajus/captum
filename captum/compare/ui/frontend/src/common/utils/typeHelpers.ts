type Literal = number | string | boolean | undefined | bigint;

const literalTypes = new Set([
  "number",
  "string",
  "boolean",
  "bigint",
  "undefined",
]);

export function isLiteral(obj: any): obj is Literal {
  return literalTypes.has(typeof obj);
}

export function isFunction(obj: any): obj is Function {
  return typeof obj === "function";
}
