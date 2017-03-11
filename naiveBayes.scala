
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.SparkSession

object movies {

  case class Sentence(sentence: String,label: Double)

  def main(args:Array[String]) {

    val spark = SparkSession
      .builder
      .appName("Movies Reviews")
      .config("spark.master", "local")
      .getOrCreate()


    // Prepare training documents from a list of (id, text, label) tuples.
    val neg = spark.sparkContext.textFile("file:///data/train/neg/").repartition(4)
      .map(w => Sentence(w, 0.0))

    val pos = spark.sparkContext.textFile("file:///data/train/pos/").repartition(4)
      .map(w => Sentence(w, 1.0))

    val test = spark.sparkContext.wholeTextFiles("file:///data/test/").repartition(4)
      .map({case(file,sentence) => (file.split("/").last.split("\\.")(0),sentence)})


    val training=neg.union(pos)
    val trainingDF=spark.createDataFrame(training)
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

    // Make predictions on test documents.
    model.transform(testDF).repartition(1)
      .select("file", "prediction")
      .write.format("csv")
      .option("header","true")
      .option("delimiter","\t")
      .save("/tmp/spark-prediction")
    spark.stop()
      }
  }
