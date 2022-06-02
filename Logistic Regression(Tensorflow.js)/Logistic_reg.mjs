//import * as tf from "@tensorflow/tfjs"
//import * as d3 from "d3"

d3.csv("./data/iris.csv").then(function(data) {
    console.log(data)
})

const load_data =  async() => {
    const csvUrl = './data/iris.csv'
    const csv_data = tf.data.csv(csvUrl, {
        columnConfigs: {
            Species: {
                isLabel: true
            }
        }
    });


    const Features = (await csv_data.columnNames()).length - 1;
    console.log(Features);
    const convert_data =
        csv_data.map(({xs, ys}) => {
            const lables = [
                ys.Species == "setosa" ? 1 : 0,
                ys.Species == "virginica" ? 1 : 0,
                ys.Species == "versicolor" ? 1 : 0
            ]
            return {
                xs: Object.values(xs),
                ys: Object.values(lables)
            };
        }).batch(10)

    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [Features],
        activation: "sigmoid",
        units: 5
    }));

    model.add(tf.layers.dense({
        activation: "softmax",
        units: 3
    }));

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.01)
    });

    await model.fitDataset(convert_data,
        {
            epochs: 150,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log("Epoch: " + epoch + "Loss: " + logs.loss);
                }
            }
        });


    const test_values = tf.tensor2d([5.8, 2.7, 5.1, 1.9], [1, 4]);

    const prediction = model.predict(test_values);
    const pIndex = tf.argMax(prediction, axis = 1).dataSync();

    const classNames = ["Setosa", "Virfinica", "Versicolor"];

    test_values.print()
    prediction.print()
    test_values.dispose();
    prediction.dispose();
    console.log(prediction);
    console.log(classNames[pIndex]);


}

load_data();
