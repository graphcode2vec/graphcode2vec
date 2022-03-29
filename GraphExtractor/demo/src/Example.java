import java.io.File; 
import java.io.IOException;
import java.util.Formatter;
import java.util.Scanner;
import java.util.Arrays;

public class Example {
	// return minimum value
	public int findMin(int[] arr) {
		int m = 0;
		for (int a : arr)
			m = Math.min(m, a);
		return m;
	}

	// return safe value
	public int getSafe(int a) {
		int m = Math.max(0, a);
		return m;
	}

	// judge if the number if prime
	private boolean isPrime(Long n) {
		if (n % 2 == 0)
			return false;
		for (int i = 3; i * i <= n; i += 2) {
			if (n % i == 0)
				return false;
		}
		return true;
	}

	// This is the method to convert byte array into hexadecimal string
	private static String toHexString(byte[] bytes) {
		Formatter formatter = new Formatter();
		for (byte b : bytes) {
			formatter.format("%02x", b);
		}
		return formatter.toString();
	}

	// Bubble Sort Algorithm
	void bubbleSort(int arr[]) {
		int n = arr.length;
		for (int i = 0; i < n - 1; i++)
			for (int j = 0; j < n - i - 1; j++)
				if (arr[j] > arr[j + 1]) {
					// swap arr[j+1] and arr[j]
					int temp = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = temp;
				}
	}

	// following Java Program ask to the user to enter a string like your first
	// name, then it will display your first name after Hello
	public void printInput() {
		String str;
		Scanner scan = new Scanner(System.in);
		System.out.print("Enter Your First Name : ");
		str = scan.nextLine();
		System.out.print("Hello, " + str);
	}

	// it will sort the array is ascending order and display the sorted array
	public void selectionSort(int arr[]) {
		int size = 0, i = 0, j = 0, temp = 0, small = 0, index = 0, count = 0;
		for (i = 0; i < (size - 1); i++) {
			small = arr[i];
			for (j = (i + 1); j < size; j++) {
				if (small > arr[j]) {
					small = arr[j];
					count++;
					index = j;
				}
			}
			if (count != 0) {
				temp = arr[i];
				arr[i] = small;
				arr[index] = temp;
			}
			count = 0;
		}
		for (i = 0; i < size; i++) {
			System.out.print(arr[i] + "  ");
		}
	}

	// create a file
	public void createFile(String filename) {
		try {
			File myObj = new File(filename);
			if (myObj.createNewFile()) {
				System.out.println("File created: " + myObj.getName());
			} else {
				System.out.println("File already exists.");
			}
		} catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
	}

	public static void countIntsize(String[] args) {
		Scanner stdIn = new Scanner(System.in);

		while (stdIn.hasNext()) {
			int a = stdIn.nextInt();
			int b = stdIn.nextInt();
			int c = a + b;
			int count = 1;

			while (c / 10 != 0) {
				count++;
				c = c / 10;
			}

			System.out.println(count);
		}
	}

	public static void sortIntList(String[] args) {
		Scanner scan = new Scanner(System.in);
		int[] heights = new int[10];
		for (int i = 0; i < 10; i++) {
			heights[i] = scan.nextInt();
		}

		Arrays.sort(heights);
		for (int i = 9; i >= 7; i--) {
			System.out.println(heights[i]);
		}
	}

	public static int lowerBound(int[] array, int length, int value) {
		int low = 0;
		int high = length;
		while (low < high) {
			final int mid = (low + high) / 2;
			if (value <= array[mid]) {
				high = mid;
			} else {
				low = mid + 1;
			}
		}
		return low;
	}

	public static int searchLowerBound(int[] array, int length, int value) {
		int low = 0;
		int high = length;
		while (low < high) {
			final int mid = (low + high) / 2;
			if (value <= array[mid]) {
				high = mid;
			} else {
				low = mid + 1;
			}
		}
		return low;
	}

	public static int findLowerBound(int[] inputs, int size, int v) {
		int bounder = 0;
		int l = size;
		int mindex = 0;
		while (bounder < l) {
			mindex = (bounder + l) / 2;
			if (v <= inputs[mindex]) {
				l = mindex;
			} else {
				bounder = mindex + 1;
			}
		}
		return bounder;
	}

	public static int getLowerBound(int v, int size, int[] inputs) {
		int h = size;
		int mindex = 0;
		int check = 0;
		while (check < h) {
			mindex = (check + h) / 2;
			if (v > inputs[mindex]) {
				check = mindex + 1;
			} else {
				h = mindex;
			}
		}
		return check;
	}

}
