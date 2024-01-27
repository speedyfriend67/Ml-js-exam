document.addEventListener('DOMContentLoaded', async () => {
  const canvas = document.getElementById('plotCanvas');
  const ctx = canvas.getContext('2d');
  const numPoints = 100;
  const data = generateData(numPoints);
  
  // Plot the data points
  tfvis.render.scatterplot(
    { name: 'Data Points' },
    { values: [data], series: ['Expressions'] },
    {
      xLabel: 'X',
      yLabel: 'Y',
      width: 500,
      height: 400,
    }
  );

  // Define the model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1], activation: 'linear' }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError', metrics: ['mse'] });

  // Train the model
  const x = tf.tensor2d(data.map(point => [point.x]));
  const y = tf.tensor2d(data.map(point => [point.y]));
  await model.fit(x, y, { epochs: 50 });

  // Plot the predictions
  plotPredictions(model, ctx);
});

function generateData(numPoints) {
  const data = [];
  for (let i = 0; i < numPoints; i++) {
    const x = Math.random() * 10;
    const y = 2 * x + Math.random() * 3; // Linear expression: y = 2x + noise
    data.push({ x, y });
  }
  return data;
}

function plotPredictions(model, ctx) {
  const [xMin, xMax] = [0, 10];
  const step = 0.1;

  ctx.beginPath();
  ctx.moveTo(xMin * 50, model.predict(tf.tensor2d([[xMin]])).dataSync()[0] * 40);
  for (let x = xMin; x <= xMax; x += step) {
    const prediction = model.predict(tf.tensor2d([[x]]));
    const y = prediction.dataSync()[0];
    ctx.lineTo(x * 50, y * 40);
  }
  ctx.stroke();
}
