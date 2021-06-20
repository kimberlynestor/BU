/*
Name: Kimberly Nestor
Class: CS 526 - Fall 2
Date: 11/10/20
Homework Problem: Homework 1 Problem 1
Description: This program has three methods, takes an array as input.
*/

import java.util.*;

public class Hw1_P1 {

    // method1 - this will find the avg, min and max of an array
    public static void stats(int[] a) {
        // initialize sum
        double sum = 0;

        // find the average of the array values
        for (int i = 0; i < a.length; i++) {
            sum = sum + a[i];
        }
        double avg = sum / a.length;

        // initialize min
        int min = a[0];
        // find the minimum value of the array
        for (int i = 1; i< a.length; i++) {
            if(a[i] < min) {
                min = a[i];
            }
        }

        // initialize max
        int max = a[0];
        // find the maximum value of the array
        for (int i = 1; i< a.length; i++) {
            if(a[i] > max) {
                max = a[i];
            }
        }

        // print out the avg, min and max
        System.out.format("\naverage = %.2f, min = %d, max = %d", avg, min, max);
    }

    // method2 - creates and prints a subarray of a given array
    public static void subarray(int[] a, int from, int to) {
        // initialize subarray
        int subarr = 0;

        // print out from and to
        System.out.format("\nThe subarray, from index %d to index %d, is: ", from, to);

        // error check w/o using Java's exception handling
        if (from < 0 || to >= a.length) {
            System.out.println("Index out of bound");
            return;
        } else {
            for (int i = from; i < to+1; i++) {
                subarr = a[i];
                // System.out.println(subarr);

                // print out the subarray
                System.out.format("\n%d ", subarr);

            }
        }
 }


    // method3 - main, calls  previous methods and prints associated values
    public static void main(String[] args) {

        // test
        int[] a = {15, 25, 10, 65, 30, 55, 65};

        System.out.println("\nGiven array is: " + Arrays.toString(a));
        stats(a);
        subarray(a,1, 4);

/*
        // test with other arrays
        int[] a = {15, 25, 10};

        System.out.println("\nGiven array is: " + Arrays.toString(a));
        stats(a);
        // subarray(a,1, 4);
        subarray(a,0, 1);
*/

    }

}
