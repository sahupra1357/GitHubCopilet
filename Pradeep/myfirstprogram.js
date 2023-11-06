function calculateDaysBetweenDates(date1, date2) {
  const differenceInMs = Math.abs(date1 - date2);
  return differenceInMs / (1000 * 60 * 60 * 24);
}
