const csvUrl = "./kc_house_data.csv";

const dataset = tf.data.csv(csvUrl, {
  columnNames: [
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
  ],
});

const dataArray = await dataset.take(5000).toArray(); // Limit to 5000 rows


const shuffledData = tf.util.shuffle(dataArray);

const OUTPUT = dataArray.map((row) => {
  return row.price;
});
const INPUT = dataArray.map((row) => [
  row.bedrooms, // Convert to an array, not an object
  row.bathrooms,
]);

const INPUTS_TENSOR = tf.tensor2d(INPUT);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUT);

function normalize(tensor) {
  return tf.tidy(() => {
    const min = tf.min(tensor, 0);
    const max = tf.max(tensor, 0);
    const normalized = tensor.sub(min).div(max.sub(min));
    return { min, max, normalized };
  });
}

const feature_results = normalize(INPUTS_TENSOR);
feature_results.normalized.print(); // Should print normalized values

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 10, activation: "relu" }));
model.add(tf.layers.dense({ units: 1 }));

model.compile({
  optimizer: tf.train.adam(),
  loss: "meanSquaredError",
});

async function trainModel() {
  await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    epochs: 50,
    batchSize: 32,
    shuffle: true,
  });
  console.log("Training complete!");
}

await trainModel();

async function predictPrice(bedrooms, bathrooms) {
  const inputTensor = tf.tensor2d([[bedrooms, bathrooms]]);

  // Normalize input using stored min/max values
  const normalizedInput = inputTensor
    .sub(feature_results.min)
    .div(feature_results.max.sub(feature_results.min));

  console.log("Normalized Input:");
  normalizedInput.print(); // Check if normalization works

  const prediction = model.predict(normalizedInput);
  console.log("Raw Prediction Tensor:");
  prediction.print(); // Should print a tensor

  const priceArray = await prediction.data();
  console.log("Extracted Prediction Array:", priceArray); // Should print array

  if (priceArray.length > 0) {
    console.log(`Predicted price: $${priceArray[0].toFixed(2)}`);
  } else {
    console.log("Prediction array is empty!");
  }
}



(async () => {
  await predictPrice(3, 2);
})();

