import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter; 
import java.util.Scanner;

public class main {
	static float[][] matrixMultiply( float A[][], float B[][]){ //fnc for matrix multiplication
		int n = A.length , m = A[0].length, l = B[0].length ;
		float result[][] = new float[n][l];
		float temp =0 ;
		for ( int i=0; i<n; i++) {
			for ( int j=0; j<l; j++) {
				for(int k=0; k<m; k++) {
					temp += A[i][k] * B[k][j] ;
				}
				result[i][j] = temp; 
				temp=0;
			}
		}
		return result;
	}

	static float[][] addMatrix(float A[][], float B[][]){ // fnc for matrix addition
		int n = A.length , m = A[0].length;
		float result[][] = new float[n][m];
		for ( int i=0; i<n; i++) 
			for ( int j=0; j<m; j++) 
				result[i][j] = A[i][j] + B[i][j]; 
		return result;
		
	}
	
	static void copyMatrix(float A[][], float B[][]) {//copy matrix B in A
		int row_num = A.length , col_num = A[0].length ;
		for(int i=0; i<row_num; i++)
			for(int j=0; j<col_num;j++)
			A[i][j] = B[i][j];
	}
	
	static void matrixToZeros(float A[][]) {//make matrix A all zeros

	int row_num = A.length , col_num = A[0].length ;
	for(int i=0; i<row_num; i++)
		for(int j=0; j<col_num;j++)
		A[i][j] = 0f;
}

     static float[][] matrixAdd_num( float[][] A,float x){ // add x to all elem in matrix A
		int n = A.length, m = A[0].length;
		float [][] result = new float[n][m];
		for ( int i=0; i<n; i++) {
			for ( int j=0; j<m; j++) { 
				result[i][j] = A[i][j] + x;
			}
		}
		return result;
	}
	
	static float[][] scalarMultiply ( float [][] A ,  float k){//mulitply all elems in A by k
		int n = A.length , m = A[0].length ;
		float[][] result = new float[n][m];
		for(int i=0; i<n;i++) {
			for (int j=0; j<m;j++) {
				result[i][j] = k * A[i][j];
			}
		}
		return result;
	}
	
	static void printMatrix(float A[][]) {//fnct to print the matrix A
		int n = A.length , m = A[0].length ;
		for ( int i=0; i<n; i++) {
			for ( int j=0; j<m; j++) {
				System.out.print(A[i][j] + " ");
			}System.out.println();
		}
	}
	
	static float L1Distance(float A[][], float B[][]) // fnc to calculate Manhattan Distance (L1 Dis) between two vectors
    {//only works for vectors of dimension Nx1
		int N = A.length ;
        float sum = 0;
        for (int i = 0; i < N; i++) 
                sum += Math.abs( A[i][0] - B[i][0]) ; 
        return sum;
    }
	
	public static void main(String[] args) throws Exception {
		float beta =  0.8f, epsilon =  0.001f ;  
	    int columns = 3, sparse_M_row_num =0; 
	    
		try{ // know number of lines in sparse Matrix in file sparseMatrix.txt
			Scanner sc = new Scanner(new BufferedReader(new FileReader("sparseMatrix_8By8.csv")));
			while(sc.hasNext()) {  
				sc.nextLine();
				sparse_M_row_num +=1 ;
				}
			sc.close();  
		}finally {} 
		
		//System.out.println(sparse_M_row_num + " ===============");
		float sparse_M[][] = new float[sparse_M_row_num][columns]; //create matrix sparse_M too save values from file
		int N= 0 ; // N should be given, could not be  calculated from sparse_M bcz the last rows in the original matrix can be null and not appear in sparse_M
		Scanner sc_in= new Scanner(System.in); //System.in is a standard input stream.
		System.out.print("Enter N (now it is 8): ");
		N= sc_in.nextInt();
		sc_in.close();
		
		float dead_end_vector[] = new float[N];//create dead_end vector to deal all-zero cols
		for(int i=0; i<N; i++)
			dead_end_vector[i]=1;
		
		try{  // reading sparse Matrix from file and filling info in sparse_M[][]
			Scanner sc = new Scanner(new BufferedReader(new FileReader("sparseMatrix_8By8.csv"))); 
		       
		      while(sc.hasNextLine()) {
		         for (int i=0; i<sparse_M.length; i++) {
		            String[] line = sc.nextLine().trim().split(",");// remove spaces and split to array line[]
		            for (int j=0; j<line.length; j++) {
		            	sparse_M[i][j] =  Float.parseFloat(line[j]); 
		            }
		            dead_end_vector[ Integer.parseInt(line[1])] =0;
		         }
		      }
		      sc.close(); 
		}
		finally {}
		   for(int i=0; i<N;i++) {
			   System.out.println(dead_end_vector[i]);
		   }
		   System.out.println("ENd of dead_end_vector");
		     
		      //Testing
		//float M[][] = {{0.5f, 0f, 0.9f, 0.1f}, {0f,0f,0f,0f}, {0.5f, 0.8f, 0f, 0f}, {0f, 0.2f, 0.1f, 0.9f} };
		 //if M is sparse and the matrix is formed by 3 columns i , j  and value
		//float sparse_M[][] = { {0f,0f,0.5f}, {0f,2f,0.9f}, {0f,3f,0.1f}, {2f,0f,0.5f}, {2f,1f,0.8f}, {3f,1f,0.2f}, {3f,2f,0.1f}, {3f,3f,0.9f} };

		
		//M = [ 0.5  0  0.9  0.1       sparse_M = [  0  0  0.5  
		//	    0    0   0   0                       0  2  0.9
		//	    0.5  0.8  0   0                      0  3  0.1
		//	    0    0.2  0.1  0.9 ]                 2  0  0.5
		//                                           2  1  0.8
		 //                                          3  1  0.2
		 //                                          3  2  0.1
		  //                                         3  3  0.9  ] 
		
		
		
		float one_nth = (float)1 / N; //the 1/n that must be filled in the initial vector
		 
		float r_t[][] = new float [N][1]; //r(t)  vector
		for(int i=0;i<N;i++)
			r_t[i][0] = one_nth;  
		float r_t_plus_one[][] = new float [N][1]; //  r(t+1)  vector
		matrixToZeros(r_t_plus_one); //r(t+1) should start as all elements are zeros and start filling elemnst from multiplication of A*r(t)
		 
		float one_min_beta_overn[][] = new float [N][1]; // (1-beta)/N
		for(int i=0;i<N;i++)
			one_min_beta_overn[i][0] = one_nth*(1-beta); 
		 
		
		//First Iteration
		for(int i=0; i<N; i++) { // add the values of 1/n 
			for(int j=0; j<N; j++) {
					r_t_plus_one[i][0]+= dead_end_vector[j] * 1/N *beta *  r_t[j][0]; // the dead_end_vector only conatins 1 for the cols that do not appear in sparse_M
				                                                       //if the col appears in sparse_M dead_end_vector will be zero and the value willnot be added
			}
		}
		
		for(int k=0; k<sparse_M_row_num; k++) //Beta . M . r
			r_t_plus_one[(int)sparse_M[k][0]] [0] += sparse_M[k][2]* beta * r_t[(int)sparse_M[k][1]][0] ;
		
		r_t_plus_one = addMatrix(r_t_plus_one, one_min_beta_overn); // + (1-beta).(1/n)e
		    
		    
	   
		while( L1Distance(r_t, r_t_plus_one) > epsilon  ) //repeat until convergence between r(t) and r(t+1) ; epsilon is determined previously
		 {  
		 copyMatrix(r_t, r_t_plus_one); // put values of r(t+1) in r(t) and r(t) become zeros to make w new iteration
		 matrixToZeros(r_t_plus_one);
		 for(int i=0; i<N; i++) { // add the values of 1/n 
				for(int j=0; j<N; j++) {
						r_t_plus_one[i][0]+= dead_end_vector[j] * 1/N *beta* r_t[j][0]; // the dead_end_vector only conatins 1 for the cols that do not appear in sparse_M
					                                                       //if the col appears in sparse_M dead_end_vector will be zero and the value willnot be added
				}
			}
		 
		 for(int k=0; k<sparse_M_row_num; k++) // multiply from sparse_M * beta by r(t) and add to values in r(t+1)
			r_t_plus_one[(int)sparse_M[k][0]] [0] += sparse_M[k][2] *beta* r_t[(int)sparse_M[k][1]][0] ; // the row in r(t) = sparse_M[k][1] 
		   
		r_t_plus_one = addMatrix(r_t_plus_one, one_min_beta_overn); // add to vector (1-beta)/N.e
		  
		 } 		 
		System.out.println("\nResults: \nThe vector r:");
 printMatrix(r_t_plus_one); 
 float sum = 0;
 for (int i=0;i<N;i++)
	 sum+=r_t_plus_one[i][0];
 System.out.println("------");
 System.out.println("The sum is: " +sum);
 System.out.println("-----");
		
			 try {  //write results to file results.txt
			      FileWriter writer = new FileWriter("results.txt"); 
			      int row_num = r_t_plus_one.length , col_num = r_t_plus_one[0].length ;
					for(int i=0; i<row_num; i++)
						for(int j=0; j<col_num;j++)
			                 writer.write(r_t_plus_one[i][j] + "\n" );
			       
			      writer.close(); // close connx
			   }
			finally{}
		 System.out.println("See results in results.txt, Thank you");
	}

}
