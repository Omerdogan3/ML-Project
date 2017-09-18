
import org.apache.log4j.{Level, Logger}




// $example on$
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
    import org.apache.spark.SparkContext._
    import org.apache.spark.SparkConf
    import org.apache.spark.sql.SQLContext


    import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression



object anothtermain {
    def main(args: Array[String]): Unit = {

      
      val spark = SparkSession.builder
      .master("local[*]")
      .appName("Example")
      .getOrCreate()
      
      

 
 val current_total_landing_count       = StructField("current_total_landing_count",       DataTypes.IntegerType)
val current_other_landing_count    = StructField("current_other_landing_count",    DataTypes.DoubleType)
val current_product_landing_count   = StructField("current_product_landing_count",   DataTypes.DoubleType)
val current_cart_landing_count    = StructField("current_cart_landing_count",    DataTypes.DoubleType)
val current_sale_amount = StructField("current_sale_amount", DataTypes.StringType)
val current_avg_cart_amount    = StructField("current_avg_cart_amount",    DataTypes.StringType)
val current_avg_visited_product_price    = StructField("current_avg_visited_product_price",    DataTypes.StringType)
val last_1_day_session_count    = StructField("last_1_day_session_count",    DataTypes.StringType)
 val last_7_day_session_count    = StructField("last_7_day_session_count",    DataTypes.StringType)
 val current_is_sale    = StructField("current_is_sale",    DataTypes.StringType)
val fields = Array(current_total_landing_count, current_other_landing_count, current_product_landing_count, current_cart_landing_count, current_sale_amount, current_avg_cart_amount,current_avg_visited_product_price,last_1_day_session_count,
    last_7_day_session_count,current_is_sale)
val schema = StructType(fields)




      val df = spark.read
                         .schema(schema)
                         .option("header", true)
                         .csv("data.csv")
df.printSchema()


val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)


val lrModel = lr.fit(df)
//learned Model Coefficients 
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")









//    val data = sc.textFile("data.csv")
//val parsedData = data.map { line =>
//  val parts = line.split(',').map(_.toDouble)
//  LabeledPoint(parts(0), Vectors.dense(parts.tail))
//}
      
      
//      
//      val maxDepth = 5
//val model = DecisionTree.train(parsedData, Classification, Gini, maxDepth)
//val labelAndPreds = parsedData.map { point =>
//  val prediction = model.predict(point.features)
//  (point.label, prediction)
//}
//val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count
//println("Training Error = " + trainErr)
//    
      
              
//  val parsedData = flight2007.map { line =>
//      val parts = line.split(',')
//      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(',').map(_.toDouble)))
//    }.cache()                  
//
//                    
//                    
// val assembler = new VectorAssembler()
//                    .setInputCols(Array("current_total_landing_count", "current_other_landing_count", "current_product_landing_count", "current_cart_landing_count", "current_sale_amount", "current_avg_cart_amount","current_avg_visited_product_price","last_1_day_session_count","last_7_day_session_count"))
//                    .setOutputCol("current_is_sale")
      
 
// val model = LinearRegressionWithSGD.train(parsedData, 100, 0.00000001)

            print("Omer Dogan::")        

      
    }
}