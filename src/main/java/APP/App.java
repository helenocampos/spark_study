/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package APP;

import model.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 *
 * @author helenocampos
 */
public class App
{

    static LogisticRegression lr;
    static LogisticRegressionModel lrModel;

    public static void main(String[] args)
    {
        Logger.getRootLogger().setLevel(Level.ERROR);
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaRegressionExample")
                .config("spark.master", "local")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        String option = "";
        Scanner sc = new Scanner(System.in);
        String modelName = "";
        while (!"0".equalsIgnoreCase(option))
        {
            System.out.println("Escolha uma opção: ");
            System.out.println("1 - Treinar modelo");
            System.out.println("2 - Carregar modelo salvo");
            System.out.println("3 - Salvar modelo atual");
            System.out.println("4 - Realizar nova inferência");
            System.out.println("5 - Inicializar inferência a partir de stream");
            System.out.println("0 - Sair");
            System.out.println("");
            option = sc.next();
            switch (option)
            {
                case "0":
                    System.out.println("Adeus");
                    break;
                case "1":
                    trainModel(spark);
                    System.out.println("Modelo treinado!");
                    break;
                case "2":
                    System.out.println("Qual o nome do modelo a ser carregado?");
                    modelName = sc.next();
                    loadModel(modelName);
                    break;
                case "3":
                    System.out.println("Qual nome deseja dar ao modelo?");
                    modelName = sc.next();
                    exportModel(modelName);
                    break;
                case "4":
                    if (lrModel != null)
                    {
                        Dataset<Row> dataToEvaluate = getEntry(spark, sc);
                        dataToEvaluate = defineFeaturesColumns(dataToEvaluate);
                        dataToEvaluate = lrModel.transform(dataToEvaluate);
                        printData(dataToEvaluate);
                    } else
                    {
                        System.out.println("Primeiro é necessário instanciar um modelo!");
                    }

                    break;

                default:
                    System.out.println("Opção inválida!");
                    break;
            }
        }

        spark.stop();
    }

    private static Dataset<Row> defineFeaturesColumns(Dataset<Row> data)
    {
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]
                {
                    "hora", "temperatura"
        })
                .setOutputCol("features");
        return assembler.transform(data);
    }

    private static void exportModel(String modelName)
    {
        try
        {
            modelName = "models/" + modelName;
            if (lrModel != null && lr != null)
            {
                lrModel.write().save(modelName + "/model");
                lr.write().save(modelName + "/lr");
                System.out.println("Modelo salvo com sucesso!");
            } else
            {
                System.out.println("Primeiro é necessário instanciar um modelo!s");
            }
        } catch (IOException ex)
        {
            java.util.logging.Logger.getLogger(App.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
    }

    private static void printData(Dataset<Row> data)
    {
        for (Row r : data.select("hora", "temperatura", "probability", "prediction").collectAsList())
        {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
                    + ", prediction=" + r.get(3));
        }
    }

    private static Dataset<Row> getEntry(SparkSession spark, Scanner sc)
    {
        System.out.println("Digite a hora no formato float. Exemplo: 18.512");
        float hora = sc.nextFloat();
        System.out.println("Digite a temperatura (inteiro). Exemplo: 20");
        int temperatura = sc.nextInt();
        return spark.createDataFrame(Arrays.asList(new MeasurementModel(hora, temperatura)), MeasurementModel.class);
    }

    private static void trainModel(SparkSession spark)
    {
        StructType customSchema = new StructType(new StructField[]
        {
            new StructField("hora", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("temperatura", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("saida", DataTypes.IntegerType, true, Metadata.empty())
        });
        Dataset<Row> training = spark.read().format("csv").option("header", "true").schema(customSchema)
                .load("dados_exemplo.csv");
        training = defineFeaturesColumns(training);
        lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.001)
                .setFeaturesCol("features")
                .setLabelCol("saida");

        lrModel = lr.fit(training);
    }

    private static void loadModel(String modelName)
    {
        File modelPath = new File("models/" + modelName);
        if (modelPath.exists())
        {
            lr = LogisticRegression.load("models/" + modelName + "/lr");
            lrModel = LogisticRegressionModel.load("models/" + modelName + "/model");
            System.out.println("Modelo carregado!");
        }else{
            System.out.println("A model with this name does not exist.");
        }
    }
}
