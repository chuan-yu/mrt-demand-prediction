import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.temporal.ChronoField;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import static mrt.Util.*;

/**
 * This programme counts the number of passengers at each MRT station
 */

public class PassengerCount {
    public static class CountMapper
            extends Mapper<LongWritable, Text, Text, Text> {

        private LocalDateTime[] dateArray;

        // In-mappers combiner
        private Map<String, Map<LocalDateTime, Integer>> countMap = new HashMap<String, Map<LocalDateTime, Integer>>();
        private int minutes;

        @Override
        protected void setup(Context context) {
            // Create an array of dates from startDate to endDate
            Configuration conf = context.getConfiguration();
            String startDate = conf.get("startDate");
            String endDate = conf.get("endDate");

            minutes = Integer.parseInt(conf.get("minutes"));
            dateArray = getDateTimeArray(startDate, endDate, minutes);

//            for (LocalDateTime dt : dateArray) {
//                System.out.println(dt.toString());
//            }

        }

        public void map(LongWritable key, Text value, Context context
        ) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();
            LocalDateTime startDate = toLocalDateTime(conf.get("startDate"), "yyyy-MM-dd HH:mm:ss");
            LocalDateTime endDate = toLocalDateTime(conf.get("endDate"), "yyyy-MM-dd HH:mm:ss");

            if (key.get() == 0 && value.toString().contains("RECORD_ID")) {
                // Skip the header of the file
                return;
            } else {
                // Get the station name and tap-in trip
                String[] splitLine = value.toString().split(",");
                String station = splitLine[9];
                String inDateTimeStr = splitLine[11];

                LocalDateTime inDateTime = toLocalDateTime(inDateTimeStr, "yyyy-MM-dd HH:mm:ss");

                if (!isBetween(inDateTime, startDate, endDate)) {
                    return;
                }

                inDateTime = inDateTime.with(ChronoField.SECOND_OF_MINUTE, 00);


                if (minutes == 1440) {
                    inDateTime = inDateTime.with(ChronoField.HOUR_OF_DAY, 00);
                    inDateTime = inDateTime.with(ChronoField.MINUTE_OF_HOUR, 00);
                } else if (minutes == 60) {
                    inDateTime = inDateTime.with(ChronoField.MINUTE_OF_HOUR, 00);
                } else {
                    int min = inDateTime.getMinute();
                    inDateTime = inDateTime.with(ChronoField.MINUTE_OF_HOUR, (min / minutes) * minutes);
                }


                // If the station does not exit in the HashMap, create an entry
                // for the station and initialize the count for each date to 0
                if (countMap.get(station) == null) {
                    Map<LocalDateTime, Integer> dateMap = new HashMap<LocalDateTime, Integer>();
                    for (int i = 0; i < dateArray.length; i++) {
                        dateMap.put(dateArray[i], 0);
                    }
                    countMap.put(station, dateMap);
                }

                // Update the count
                //System.out.println(inDateTime);
                int currentCount = countMap.get(station).get(inDateTime);
                countMap.get(station).put(inDateTime, currentCount + 1);
            }
        }

        @Override
        protected void cleanup(Context context)
                throws IOException, InterruptedException {

            // Spill the HashMap records to the disk in the form (station, date:count)
            for (Map.Entry<String, Map<LocalDateTime, Integer>> entry: countMap.entrySet()) {
                String station = entry.getKey();
                Map<LocalDateTime, Integer> valueMap = entry.getValue();

                for (Map.Entry<LocalDateTime, Integer> item: valueMap.entrySet()) {
                    String date = item.getKey().toString();
                    int count = item.getValue();
                    Text outputKey = new Text(station);
                    Text outputValue = new Text(date + "=" + Integer.toString(count));
                    context.write(outputKey, outputValue);

                }
            }
        }
    }

    public static class CountReducer
            extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {



            // TreeMap to store the count by date
            Map<LocalDateTime, Long> countByDate = new TreeMap<LocalDateTime, Long>();

            for (Text val : values) {
                // Get station and date
                String[] splitVal = val.toString().split("=");
                String dateTimeStr = splitVal[0];
                long count = Long.parseLong(splitVal[1]);
                LocalDateTime date = toLocalDateTime(dateTimeStr, "yyyy-MM-dd'T'HH:mm");

                // Update the count
                if (countByDate.get(date) == null) {
                    countByDate.put(date, count);
                } else {
                    long currentCount = countByDate.get(date);
                    countByDate.put(date, currentCount + count);
                }
            }

            // Spill the TreeMap records to disk
            String outValue = "";
            for (Map.Entry<LocalDateTime, Long> entry: countByDate.entrySet()) {
                String outDate = entry.getKey().toString();
                String outCount = entry.getValue().toString();
//                outValue += outDate + "=" + outCount + " ";
                outValue += outCount + ",";
            }
            context.write(key, new Text(outValue));

        }

    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("mapred.textoutputformat.separator", ",");

        // Define the start and end date
        String startDate = "2016-03-01 00:00:00";
        String endDate = "2016-03-31 23:59:59";

        conf.set("startDate", startDate);
        conf.set("endDate", endDate);

        String period = args[2];
        System.out.println(period);

        if (period.equals("day")) {
            conf.set("minutes", "1440");

        } else if (period.equals("hour")) {
            conf.set("minutes", "60");

        } else if (period.equals("quarter")) {
            conf.set("minutes", "15");

        } else {
            throw new IllegalArgumentException("invalid period value");
        }

        Job job = new Job(conf, "passenger count by hour");
        job.setJarByClass(PassengerCount.class);
        job.setMapperClass(CountMapper.class);
        job.setReducerClass(CountReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}