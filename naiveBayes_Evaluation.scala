/**
 * Created by eva on 27/02/2017.
 */
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.SparkSession

object movies {

  case class TrainSentence(file: String,sentence: String)
  case class Sentence(sentence: String,label: Double)

  def main(args:Array[String]) {


    val spark = SparkSession
      .builder
      .appName("Movies Reviews")
      .config("spark.master", "local")
      .getOrCreate()


    // Prepare training documents from a list of (id, text, label) tuples.
    val neg = spark.sparkContext.textFile("file:///Users/eva/Desktop/Master/TechnologiesforBigDataManagementandAnalytics/MovieReviews_assignment/data/train/neg/").repartition(4)
      .map(w => Sentence(w, 0.0))

    val pos = spark.sparkContext.textFile("file:///Users/eva/Desktop/Master/TechnologiesforBigDataManagementandAnalytics/MovieReviews_assignment/data/train/pos/").repartition(4)
      .map(w => Sentence(w, 1.0))

    val test = spark.sparkContext.wholeTextFiles("file:///Users/eva/Desktop/Master/TechnologiesforBigDataManagementandAnalytics/MovieReviews_assignment/data/test/").repartition(4)
      .map({case(file,sentence) => TrainSentence(file.split("/").last.split("\\.")(0),sentence)})


    val training=neg.union(pos)
    val trainingDF=spark.createDataFrame(training).toDF("sentence","label")
    val testDF=spark.createDataFrame(test)

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and Naive Bayes
    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val nb = new NaiveBayes()

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, nb))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(trainingDF)

    // Prepare test documents, which are unlabeled (id, text) tuples.
    val Array(subTrainDF, subTestDF) = trainingDF.randomSplit(Array(0.6, 0.4))

    // train using sub-training set (60%)
    val evalModelNB = pipeline.fit(subTrainDF)
    // predict on sub-test set (40%)
    val evalPredNB = evalModelNB.transform(subTestDF)

    // create sql view
    evalPredNB.createOrReplaceTempView("docs")

    // sql queries to gather stats
    evalPredNB.sqlContext.sql("select count(*) as total_pos from docs where label == 1.0").show()
    evalPredNB.sqlContext.sql("select count(*) as total_neg from docs where label == 0.0").show()
    evalPredNB.sqlContext.sql("select count(*) as true_pos from docs where label == 1.0 and label==prediction").show()
    evalPredNB.sqlContext.sql("select count(*) as true_neg from docs where label == 0.0 and label==prediction").show()


    // Make predictions on test documents.
    val predictionNB=model.transform(testDF)

    val binaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction")
  val areaUnderROC=binaryClassificationEvaluator.setMetricName("areaUnderROC").evaluate(evalPredNB)
  val areaUnderPR = binaryClassificationEvaluator.setMetricName("areaUnderPR").evaluate(evalPredNB)

    println("The areaUnderROC is "+areaUnderROC)
    println("The areaUnderPR is "+ areaUnderPR )

      predictionNB.repartition(1)
      .select("file", "prediction")
      .write.format("csv")
      .option("header","true")
      .option("delimiter","\t")
      .save("/Users/eva/Desktop/spark-master/NaiveBayes_Evaluation/spark-prediction")
    spark.stop()
  }
}
