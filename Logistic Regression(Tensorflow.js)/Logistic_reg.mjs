//import * as tf from "@tensorflow/tfjs";
//import * as tfvis from "@tensorflow/tfjs-vis";
//import * as d3 from "d3";
//import Plotly from "plotly.js-dist-min";


const load_data = async () => {
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
    console.log(convert_data);

    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [Features],
                activation: "sigmoid",
                units: 5
            }),
            tf.layers.dense({
                activation: "softmax",
                units: 3
            })
        ]
    });

    const trainLogs = [];
    const lossContainer = document.getElementById("loss-count");
    const accContainer = document.getElementById("acc-count");

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.01),
        metrics: ["accuracy"]
    });

    function onBatchEnd(batch, logs) {
        console.log('Accuracy', logs.acc);
    }

    model.summary();




    const surface = {name: 'show.fitCallbacks', tab: 'Training'};


    await model.fitDataset(convert_data,
        {
            epochs: 120,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    trainLogs.push(logs);
                    tfvis.show.history(lossContainer, trainLogs, ['loss']);
                    tfvis.show.history(accContainer, trainLogs, ['acc']);
                }
            }

        });
    
    // await model.save('downloads://Logistic_regression')


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
