package mrt;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;


public class Util {

    /**
     * Convert a string to date
     * @param date      date string
     * @param pattern   pattern of the date
     * @return          date
     */

    public static LocalDate toLocalDate(String date, String pattern) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(pattern);
        return LocalDate.parse(date, formatter);
    }

    public static LocalDateTime toLocalDateTime(String date, String pattern) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(pattern);
        return LocalDateTime.parse(date, formatter);
    }

    /**
     * Returns an Array of dates between the start and end date (inclusive)
     *
     * @param start the start date of format "yyyy-MM-dd"
     * @param end   the end date of format "yyyy-MM-dd"
     * @return      an array of dates
     */
    public static LocalDate[] getDateArray(String start, String end) {
        // convert strings to dates
        LocalDate startDate = toLocalDate(start, "yyyy-MM-dd");
        LocalDate endDate = toLocalDate(end, "yyyy-MM-dd");

        // create an array of dates
        long daysBetween = ChronoUnit.DAYS.between(startDate, endDate);
        LocalDate[] dateArray = new LocalDate[(int) daysBetween + 1];
        int i = 0;
        for (LocalDate date = startDate; date.isBefore(endDate.plusDays(1)); date=date.plusDays(1)) {
            dateArray[i] = date;
            i++;
        }

        return dateArray;
    }


    public static LocalDateTime[] getDateTimeArray(String start, String end, int minutes) {
        LocalDateTime startDateTime = toLocalDateTime(start, "yyyy-MM-dd HH:mm:ss");
        LocalDateTime endDateTime = toLocalDateTime(end, "yyyy-MM-dd HH:mm:ss");

        long len = ChronoUnit.MINUTES.between(startDateTime, endDateTime);
        len = len / minutes;
        LocalDateTime[] dateTimeArray = new LocalDateTime[(int) len + 1];
        int i = 0;
        for (LocalDateTime dateTime = startDateTime; dateTime.isBefore(endDateTime); dateTime=dateTime.plusMinutes(minutes)){
            dateTimeArray[i] = dateTime;
            i++;
        }
        return dateTimeArray;
    }


    public static boolean isBetween(LocalDateTime dateTime, LocalDateTime startTime, LocalDateTime endTime) {
        return ((dateTime.equals(startTime) || dateTime.isAfter(startTime)) &&
                (dateTime.equals(endTime) || dateTime.isBefore(endTime)));
    }
}

