/*
Name: Kimberly Nestor
Class: CS 526 - Fall 2
Date: 12/08/20
Homework Problem: Homework 5 Problem 7
Description: This program does a test to estimate the running time of sorting
algorithms: insertionsort, mergesort, quicksort and heapsort.
*/

import java.util.*;
import java.lang.Math;
import java.lang.Integer;


public class Hw5_P7 {

    // sort random numbers using INSERTIONSORT from pg111 of textbook, slightly modified
    public static void insertionSort(int[] data) {
        int n = data.length;
        for (int k = 1; k < n; k++) {
            int cur = data[k];
            int j = k;
            while (j > 0 && data[j - 1] > cur) {
                data[j] = data[j - 1];
                j--;
            }
            data[j] = cur;
        }
    }

    // merge operation for a java array from pg 537 of textbook, slightly modified
    public static void merge(int[] S1, int[] S2, int[] S) {
        int i = 0, j = 0;
        while (i + j < S.length) {
            if (j == S2.length || (i < S1.length && Integer.compare(S1[i], S2[j]) < 0))
                S[i + j] = S1[i++]; // copy ith element of S1 and increment i
            else
                S[i + j] = S2[j++]; // copy jth element of S2 and increment j
        }
    }

    // sort random numbers using MERGESORT from pg538 of textbook, slightly modified
    public static void mergeSort(int[] S) {
        int n = S.length;
        if (n < 2) return;
        // divide
        int mid = n / 2;
        int[] S1 = Arrays.copyOfRange(S, 0, mid);
        int[] S2 = Arrays.copyOfRange(S, mid, n);
        // conquer (with recursion)
        mergeSort(S1);
        mergeSort(S2);
        // merge results
        merge(S1, S2, S);
    }

    // QUICKSORT from baeldung - https://www.baeldung.com/java-quicksort
    // recursive quick sort method
    public static void quickSort(int arr[], int begin, int end) {
        if (begin < end) {
            int partitionIndex = partition(arr, begin, end);
            quickSort(arr, begin, partitionIndex - 1);
            quickSort(arr, partitionIndex + 1, end);
        }
    }

    // partition helper method for recursive quicksort - from baeldung (above)
    private static int partition(int arr[], int begin, int end) {
        int pivot = arr[end];
        int i = (begin - 1);
        for (int j = begin; j < end; j++) {
            if (arr[j] <= pivot) {
                i++;
                int swapTemp = arr[i];
                arr[i] = arr[j];
                arr[j] = swapTemp;
            }
        }
        int swapTemp = arr[i + 1];
        arr[i + 1] = arr[end];
        arr[end] = swapTemp;
        return i + 1;
    }

    // HeapSort from geeksforgeeks - https://www.geeksforgeeks.org/heap-sort/
    public static void heapSort(int arr[]) {
        int n = arr.length;

        // Build heap (rearrange array)
        for (int i = n / 2 - 1; i >= 0; i--)
            heapify(arr, n, i);

        // One by one extract an element from heap
        for (int i = n - 1; i > 0; i--) {
            // Move current root to end
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;

            // call max heapify on the reduced heap
            heapify(arr, i, 0);
        }
    }

    // To heapify a subtree rooted with node i which is
    // an index in arr[]. n is size of heap
    static void heapify(int arr[], int n, int i) {
        int largest = i; // Initialize largest as root
        int l = 2 * i + 1; // left = 2*i + 1
        int r = 2 * i + 2; // right = 2*i + 2

        // If left child is larger than root
        if (l < n && arr[l] > arr[largest])
            largest = l;

        // If right child is larger than largest so far
        if (r < n && arr[r] > arr[largest])
            largest = r;

        // If largest is not root
        if (largest != i) {
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;

            // Recursively heapify the affected sub-tree
            heapify(arr, n, largest);
        }
    }


    public static void runTimeExp() {

        int n = 10000; //10000
        // for loop runs sort using algorithms ten times, using n = 10k - 100k
        for (int i = 0; i < 10; i++) {

            // initialize the following variables
            int[] sortNum = new int[n];
            int[] sortNumCopy;
            long startTime, endTime, elapsedTime;

            // loop creates array of n random integers from 1-1mil
            for (int ii = 0; ii < n; ii++) {
                // generate random numbers within range 1-1mil
                int rand_num = (int) (Math.random() * 1000000) + 1;
                sortNum[ii] = rand_num;
//                System.out.println(rand_num);
            }

            // clone unsorted array
            sortNumCopy = sortNum.clone();
            // start time for INSERTIONSORT
            startTime = System.currentTimeMillis();
            // call insertionSort on copy of array
            insertionSort(sortNumCopy);
            // end time for INSERTIONSORT
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            System.out.format("InsertionSort time (n=%d) = %d\n", n, elapsedTime);
//            System.out.println(Arrays.toString(sortNumCopy));

            // clone unsorted array
            sortNumCopy = sortNum.clone();
            // start time for MERGESORT
            startTime = System.currentTimeMillis();
            // call mergeSort on copy of array
            mergeSort(sortNumCopy);
            // end time for MERGESORT
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            System.out.format("MergeSort time (n=%d) = %d\n", n, elapsedTime);
//            System.out.println(Arrays.toString(sortNumCopy));

            // clone unsorted array
            sortNumCopy = sortNum.clone();
            // start time for QUICKSORT
            startTime = System.currentTimeMillis();
            // call quickSort on copy of array
            quickSort(sortNumCopy, 0, sortNumCopy.length-1);
            // end time for QUICKSORT
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            System.out.format("QuickSort time (n=%d) = %d\n", n, elapsedTime);
//            System.out.println(Arrays.toString(sortNumCopy));

            // clone unsorted array
            sortNumCopy = sortNum.clone();
            // start time for HEAPSORT
            startTime = System.currentTimeMillis();
            // call heapSort on copy of array
            heapSort(sortNumCopy);
            // end time for HEAPSORT
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            System.out.format("HeapSort time (n=%d) = %d\n", n, elapsedTime);
//            System.out.println(Arrays.toString(sortNumCopy));



//            System.out.println(n);
            // add 10k to n after each iteration of the loop
            n += 10000; //10000
//            System.out.println(sortNum.length);
//            System.out.println(Arrays.toString(sortNum));

        }
    }

    public static void main(String[] args) {
        runTimeExp();
    }
}



//substantive comments

