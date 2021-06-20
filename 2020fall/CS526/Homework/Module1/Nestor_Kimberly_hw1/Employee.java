/*
Name: Kimberly Nestor
Class: CS 526 - Fall 2
Date: 11/10/20
Homework Problem: Homework 1 Problem 2
Description: This program contains an employee class.
*/

public class Employee {
	// instance variables
	private int empId;
	private String name;
	
	// constructor
	public Employee() { }
	
	public Employee(int empId, String name) {
		this.empId = empId;
		this.name = name;
	}
	
	// get methods
	public int getEmpId() { return empId;}
	public String getName() { return name;}
	
	// set methods - Input argument: empId of integer type
	public void setEmpId(int empId) {
		this.empId = empId;
	}

	// set methods - Input argument: name of String type
	public void setName(String name) {
		this.name = name;
	}
}
