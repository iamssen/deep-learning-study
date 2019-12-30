console.log((1/3) * (Math.pow(1 * 1 - 1, 2) + Math.pow(1 * 2 - 2, 2) + Math.pow(1 * 3 - 3, 2)));
console.log((1/3) * (Math.pow(0 * 1 - 1, 2) + Math.pow(0 * 2 - 2, 2) + Math.pow(0 * 3 - 3, 2)));
console.log((1/3) * (Math.pow(2 * 1 - 1, 2) + Math.pow(2 * 2 - 2, 2) + Math.pow(2 * 3 - 3, 2)));

function cost(W, data) {
  return (1 / data.length) * data.map(({x, y}) => Math.pow(W * x - y, 2)).reduce((sum, i) => sum + i, 0);
}

const data = [
  {x: 1, y: 1}, 
  {x: 2, y: 2}, 
  {x: 3, y: 3},
];

console.log(cost(1, data));
console.log(cost(0, data));
console.log(cost(2, data));