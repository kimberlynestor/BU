/*
Name: Kimberly Nestor
Class: CS 526 - Fall 2
Date: 12/01/20
Homework Problem: Homework 4 Problem 6
Description: This program does a test to estimate the running time of data structures
Map, ArrayList and LinkedList.
*/

import java.util.*;
import java.lang.Math;


public class Hw4_P6 {

    public static void runTimeExp() {
        // initiate the following variables
        double totExpTime_map = 0;
        double avgExpTime_map = 0;
        double totExpTime_arrayList = 0;
        double avgExpTime_arrayList = 0;
        double totExpTime_linkedList = 0;
        double avgExpTime_linkedList = 0;

        double totExpTime_mapSearch = 0;
        double avgExpTime_mapSearch = 0;
        double totExpTime_arrayListSearch = 0;
        double avgExpTime_arrayListSearch = 0;
        double totExpTime_linkedListSearch = 0;
        double avgExpTime_linkedListSearch = 0;

        int[] searchKeys = new int[100000];

        // for loop runs insertion and search into data structures ten times
        for (int i = 0; i < 10; i++) {

            // loop creates array of 100k random integers from 1-1mil
            //int count=0;
            //int[] insertKeys = new int[10];
            int[] insertKeys = new int[100000];
            for (int ii = 0; ii < 100000; ii++) { //100000
                // generate random numbers within range 1-1mil
                int rand_num = (int) (Math.random() * 1000000) + 1;
                insertKeys[ii] = rand_num;
                //System.out.println(rand_num);
            }
            //System.out.println(Arrays.toString(insertKeys));

            // start time for HASHMAP INSERTION
            long startTime, endTime, elapsedTime;
            startTime = System.currentTimeMillis();
            // insert keys from insertKeys into HashMap
            HashMap<Integer, Integer> myMap = new HashMap<Integer, Integer>();
            for (int j = 0; j < insertKeys.length; j++) {
                myMap.put(insertKeys[j], 0);
            }
            //System.out.println(myMap);
            //System.out.println(myMap.size());
            // end time for hashMap insertion
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            totExpTime_map += elapsedTime;


            // start time for ARRAYLIST INSERTION
            startTime = System.currentTimeMillis();
            // insert keys from insertKeys into ArrayList
            ArrayList<Integer> myArrayList = new ArrayList<Integer>();
            for (int jj = 0; jj < insertKeys.length; jj++) {
                myArrayList.add(insertKeys[jj]);
            }
//            System.out.println(myArrayList);
//            System.out.println(myArrayList.size());
            // end time for arraylist insertion
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            totExpTime_arrayList += elapsedTime;


            // start time for LINKEDLIST INSERTION
            startTime = System.currentTimeMillis();
            // insert keys from insertKeys into LinkedList
            LinkedList<Integer> myLinkedList = new LinkedList<Integer>();
            for (int jjj = 0; jjj < insertKeys.length; jjj++) {
                myLinkedList.add(insertKeys[jjj]);
            }
//            System.out.println(myLinkedList.size());
            // end time for LinkedList insertion
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            totExpTime_linkedList += elapsedTime;
            //System.out.println(myLinkedList);
            //System.out.println(totExpTime_linkedList);


            // loop creates array of 100k random integers from 1-2mil
            //int[] searchKeys = new int[10];
            for (int iii = 0; iii < 100000; iii++) { //100000
                // generate random numbers within range 1-2mil
                int rand_num = (int) (Math.random() * 2000000) + 1;
                searchKeys[iii] = rand_num;
                //System.out.println(rand_num);
            }
            //System.out.println(Arrays.toString(insertKeys));


            // start time for HASHMAP SEARCH
            startTime = System.currentTimeMillis();
            //for each key in searchKeys check to see if in map
            for (int l = 0; l < searchKeys.length; l++) {
                // search keys from searchKeys to see if in HashMap
                myMap.containsKey(searchKeys[l]);
                //System.out.println(myMap.containsKey(searchKeys[l]));
            }
            // end time for hashMap search
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            totExpTime_mapSearch += elapsedTime;

            // start time for ARRAYLIST SEARCH
            startTime = System.currentTimeMillis();
            //for each key in searchKeys check to see if in arraylist
            for (int ll = 0; ll < searchKeys.length; ll++) {
                // search keys from searchKeys to see if in arraylist
                myArrayList.contains(searchKeys[ll]);
                //System.out.println(myArrayList.containsKey(searchKeys[ll]));
            }
            // end time for arraylist search
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            totExpTime_arrayListSearch += elapsedTime;

            // start time for LINKEDLIST SEARCH
            startTime = System.currentTimeMillis();
            //for each key in searchKeys check to see if in linkedlist
            for (int lll = 0; lll < searchKeys.length; lll++) {
                // search keys from searchKeys to see if in linkedlist
                myLinkedList.contains(searchKeys[lll]);
                //System.out.println(myLinkedList.containsKey(searchKeys[lll]));
            }
            // end time for linkedlist search
            endTime = System.currentTimeMillis();
            elapsedTime = endTime - startTime;
            totExpTime_linkedListSearch += elapsedTime;

        }

        //number of keys
        System.out.format("Number of keys = " + searchKeys.length);
        System.out.println("\n");

        // average estimated running time for HashMap insertion
        avgExpTime_map = totExpTime_map/10;
        System.out.format("HashMap average total insert time = " + avgExpTime_map);
        // average estimated running time for arrayList insertion
        avgExpTime_arrayList = totExpTime_arrayList/10;
        System.out.format("\nArrayList average total insert time = " + avgExpTime_arrayList);
        // average estimated running time for LinkedList insertion
        avgExpTime_linkedList = totExpTime_linkedList/10;
        System.out.format("\nLinkedList average total insert time = " + avgExpTime_linkedList);
        System.out.println("\n");

        // average estimated running time for HashMap search
        avgExpTime_mapSearch = totExpTime_mapSearch/10;
        System.out.format("HashMap average total search time = " + avgExpTime_mapSearch);
        // average estimated running time for arrayList search
        avgExpTime_arrayListSearch = totExpTime_arrayListSearch/10;
        System.out.format("\nArrayList average total search time = " + avgExpTime_arrayListSearch);
        // average estimated running time for linkedList search
        avgExpTime_linkedListSearch = totExpTime_linkedListSearch/10;
        System.out.format("\nLinkedList average total search time = " + avgExpTime_linkedListSearch);

    }

    public static void main(String[] args) {
        runTimeExp();
    }
}



//substantive comments
// for insertion arraylist is the shortest, and linkedlist takes twice as long as arraylist
// for insertion the hashmap is the longest and takes about 4x as long as arraylist

// hashmap does a very quick search
// arraylist takes FOREVER, much longer than hashmap took to insert
// linkedlist search is three time longer than arraylist search,
// exponentially longer than the hashmap search



