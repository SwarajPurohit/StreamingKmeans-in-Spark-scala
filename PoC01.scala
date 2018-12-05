import org.apache.spark._
import org.apache.spark.mllib.clustering.StreamingKMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint

sc.setLogLevel("OFF")
sc.stop

val conf = new SparkConf().setMaster("local[2]").setAppName("ExampleStreamingKmeans")
val ssc = new StreamingContext(conf, Seconds(3))

val data_train = ssc.textFileStream("train.txt").map(Vectors.parse)
val data_test = ssc.textFileStream("test.txt").map(LabeledPoint.parse)

val model = new StreamingKMeans()
model.setDecayFactor(0.5)
model.setK(3)
model.setRandomCenters(2, 0.0)

model.trainOn(data_train)
model.predictOnValues(data_test.map(lp => (lp.label, lp.features))).print()


ssc.start()
ssc.awaitTerminationOrTimeout(10)
