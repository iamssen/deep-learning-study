/*
sigmoid(0) === 0.5 를 기준으로
값이 한없이 떨어져도 0 미만으로 떨어지지도 않고,
값이 한없이 올라가도 1 이상으로 올라가지도 않는 function

sigmoid(x) > 0.5 를 사용해서 boolean 값을 잡을때 사용할 수 있을듯 싶다
*/

function sigmoid(t) {
  return 1 / (1 + Math.exp(-t));
  // return 1 / (1 + Math.pow(Math.E, -t))
}

let f = -15;
const fmax = -f;

while (f < fmax) {
  console.log(f, sigmoid(f));
  f += 1;
}

console.log(Number.MIN_SAFE_INTEGER, sigmoid(Number.MIN_SAFE_INTEGER));
console.log(Number.MAX_SAFE_INTEGER, sigmoid(Number.MAX_SAFE_INTEGER));