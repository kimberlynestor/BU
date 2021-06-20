/*
Name: Kimberly Nestor
Class: CS 526 - Fall 2
Date: 11/10/20
Homework Problem: Homework 1 Problem 2
Description: This program contains the subclass salaried employee.
*/


public class SalariedEmployee extends Employee {
    // Instance variable
    private double salary;

    //get method - allows you to get private salary from another module
    public double getSalary() {
        return salary;
    }

    //set method - set salary from another moduule
    public void setSalary(double salary) {
        this.salary = salary;
    }

    // constructor - constructs the cloass
    SalariedEmployee(int empId, String name, double salary) {
        super(empId, name);
        this.salary = salary;
    }


    // employee monthly payment method - what each employee gets paid per month
    public double monthlyPayment() {
        double monthly_sal = this.salary / 12;
        return monthly_sal;
    }

    // prints out all the employee info: ID, name, annual salary, monthly salary
    public void employeeInfo() {
//        System.out.println(Employee.getEmpId());
        System.out.println(getEmpId());
        System.out.println(getName());
        System.out.println(this.salary);
        System.out.println(monthlyPayment());

    }

    public static void main(String[] args) {

        SalariedEmployee person_var = new SalariedEmployee(25252, "Kim", 288000);
        person_var.employeeInfo();
//        employeeInfo();

//        Employee person_var2 = new Employee(25252, "Kim");
//        System.out.println(person_var2);
    }


}



