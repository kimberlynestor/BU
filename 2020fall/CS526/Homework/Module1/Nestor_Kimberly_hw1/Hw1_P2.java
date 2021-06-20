/*
Name: Kimberly Nestor
Class: CS 526 - Fall 2
Date: 11/10/20
Homework Problem: Homework 1 Problem 2
Description: This program takes an input of employee information, calls the
SalariedEmployee and prints employe information within a salaary threshold.
*/

import java.util.ArrayList;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Hw1_P2 {

    // Reads employes from an input file and stores them in ArrayList
    public static void employeesAbove(ArrayList<SalariedEmployee> empList) throws FileNotFoundException {

        // read the employee file
        Scanner empListScanner = new Scanner(new File("employee_input.txt"));
        // input args for empID, name and salary
        String aEmp;
        String empId;
        String name;
        String salary;
        //
        int monthly;
        int threshold = 70000;

        // read the file and store the entries in a list
        while (empListScanner.hasNext()) {
            aEmp = empListScanner.nextLine();
            Scanner empScanner = new Scanner(aEmp).useDelimiter(", ");

            // separate arguments ID, name and salary from each string line by ,
            empId = empScanner.next();
            int empId2 = Integer.parseInt(empId);
            name = empScanner.next();
            salary = empScanner.next();
            double salary2 = Integer.parseInt(salary);

            // if statement to determine which employees are in the threshold
            if(salary2 > threshold) {
                System.out.println(empId);
                System.out.println(name);
                System.out.println(salary);

//            System.out.println(empList);
            }
            empScanner.close();
        }
        empListScanner.close();
    }


    public static void main(String[] args) {
        // check that there are no hard crashes and if not print out employees in threshold
        ArrayList<SalariedEmployee> empList = new ArrayList();
        try {
            employeesAbove(empList);
        }
        catch (FileNotFoundException e){
            System.out.println("Input file not found.");
        }
//
//        System.out.println("Print all employees");
//        System.out.println("The number of employees in the list = " + empList.size());
//        printAllEmps(empList);
//        System.out.println();

    }
}
