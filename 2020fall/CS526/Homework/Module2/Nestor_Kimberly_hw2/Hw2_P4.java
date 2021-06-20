/*
Name: Kimberly Nestor
Class: CS 526 - Fall 2
Date: 11/17/20
Homework Problem: Homework 1 Problem 4
Description: This program contains two recursive functions that 1) reverse a
subarray and 2) arranges even numbers before odd
*/


import java.util.Arrays;

public class Hw2_P4 {
	// this is a recursive method that will take an array and print subarray in
	// reverse order
	public static void ReverseHelp(int[] array, int start, int end) {
		// check conditions
		if (start < end) {
			// rearrange around indices and return the array
			int temp = array[start];
			array[start] = array[end];
			array[end] = temp;
			// call recursive and set new array
			ReverseHelp(array, start+1, end-1);
		}
	}

	// this is a helper method - used with ReverseHelp() to ensure fewer input args
	public static void reverseFirstN(int[] a, int n) {
		ReverseHelp(a, 0, n);
	}

	// this is a helper method for EvenOddHelp() to allow for one arg
	public static void evenBeforeOdd(int[] a) {
		EvenOddHelp(a, a.length);
	}

	// this is the recursive method used to separate the even and odd values
	public static void EvenOddHelp(int[] array, int n) {
		if (n == 0) {
			return;
		} else if (array[n - 1] % 2 == 0) {
			// separate values according to remainer =1 or !=1
			for (int i = 0; i < n - 1; i++) {
				if (array[i] % 2 != 0) {
					// sort list into even and odd
					int temp = array[i];
					array[i] = array[n - 1];
					array[n - 1] = temp;
					// call even odd separation recursively
					EvenOddHelp(array, n - 1);
				}
			}
		} else {
			EvenOddHelp(array, n - 1);
		}
	}

	// MAIN section - used to run the above methods and print in terminal
	public static void main(String[] args) {
		// initialize and fill array
		int[] a = new int[10];
		for (int i = 0; i < a.length; i++) {
			a[i] = (i + 1) * 10;
		}

		System.out.println("Initial array: ");
		System.out.println(Arrays.toString(a));
		System.out.println();


		// test section
		/*
		System.out.println("TEST = ");
		reverseFirstN(intArrayCopy, 5);
		System.out.println(Arrays.toString(intArrayCopy));
		*/

		// initialize and clone testing array
		int[] intArrayCopy;
		intArrayCopy = a.clone();

		int N = 2;


		// reverse subarray section
		reverseFirstN(intArrayCopy, N);
		System.out.println("\nAfter reversing first " + N + " elements: ");
		System.out.println(Arrays.toString(intArrayCopy));
		System.out.println();

		intArrayCopy = a.clone();
		N = 7;
		reverseFirstN(intArrayCopy, N);
		System.out.println("\nAfter reversing first " + N + " elements: ");
		System.out.println(Arrays.toString(intArrayCopy));
		System.out.println();

		// evenBeforeOdd() method section
		int[] b = {10, 15, 20, 30, 25, 35, 40, 45};
//		int[] b = {11, 15, 14, 313, 20, 35, 53, 45};

		System.out.println("\nBefore rearrange: ");
		System.out.println(Arrays.toString(b));
		System.out.println();

		evenBeforeOdd(b);
		System.out.println("\nAfter rearrange: ");
		System.out.println(Arrays.toString(b));
		System.out.println();

		// test section
		/*
		EvenOddHelp(b, b.length);
		System.out.println("TEST = ");
		System.out.println(Arrays.toString(b));
		System.out.println();
		*/
	}
}
