# connect to oracle database
import java.sql.*;
public class firstjavacode {
public static void main(String[] args) {
try {
// Load the driver
Class.forName("oracle.jdbc.driver.OracleDriver");
// Connect to the database
Connection con = DriverManager.getConnection("jdbc:oracle:thin:@localhost:1521:xe", "system", "manager");
// Create a Statement
Statement stmt = con.createStatement();
// Execute a query
ResultSet rs = stmt.executeQuery("select * from emp");
// Iterate through the result and print the employee names
while (rs.next())
System.out.println(rs.getString(1));

// Close the connection
con.close();
}
catch (Exception e) {
System.out.println(e);
}
}
}
