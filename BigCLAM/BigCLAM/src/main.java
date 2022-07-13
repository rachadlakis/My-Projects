import java.io.*;  
import java.util.Random;

public class main {
	static double[][] matrixMultiply( double A[][], double B[][]){ //fnc for matrix multiplication
		int n = A.length , m = A[0].length, l = B[0].length ;
		double result[][] = new double[n][l];
		double temp =0 ;
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
	
	static void printMatrix(double A[][]) {//fnct to print the matrix A
		int n = A.length , m = A[0].length ;
		for ( int i=0; i<n; i++) {
			for ( int j=0; j<m; j++) {
				System.out.print(A[i][j] + "||");
			}System.out.println();
		}
	}
	static double L1Distance(double A[], double B[]) // fnc to calculate Manhattan Distance (L1 Dis) between two vectors
    {//only works for vectors of dimension Nx1
		int N = A.length ;
        double sum = 0;
        for (int i = 0; i < N; i++) 
                sum += Math.abs( A[i] - B[i]) ; 
        return sum;
    }

	static double scalar_product(double A[][], int i, int j) {
		double result=0.0;
	     int width = A[0].length;
	     for(int l=0; l<width; l++) {
	    	 result+= (A[i][l] * A[j][l]);
	    	 
	     }return result;

	}
	static void fillZeros(double A[]) {
		for (int i=0; i<A.length;i++)
			A[i]=0.0;
	}
	
	public static void main(String[] args) throws IOException {
		 String temp1[] = new String[3];
		int file_length=0 , nodes_number=0, Communities_num=3; //the file length is not as mentioned 713319 but 486564
		String source = "soc-twitter-follows2.txt";
		double eta_reduction_factor = 0.9, eta =0.1;//if thousands of nodes, -SUm Fv becomes >>> the rest, the results -> negative -> zeros
		String line ="";
		
		try {
			File file = new File(source);
			FileReader fr = new FileReader(file);
			BufferedReader br=new BufferedReader(fr);  //creates a buffering character input stream  
			br.readLine(); 
			line =br.readLine();
			temp1 = line.trim().split(" ");
			file_length = Integer.parseInt(temp1[2]);
			nodes_number = Integer.parseInt(temp1[0]); 
			
			fr.close();
			}
			catch(IOException e) {}
//		System.out.println("nodes_number = " + nodes_number);
//		System.out.println("file_length = " + file_length);

		Random rand = new Random();
		
		
		double CMS_Matrix[][] = new double[nodes_number][Communities_num]; // what should be the number of communities??
		int original_matrix[][] = new int[file_length][2];
		double random;
		for (int m=0; m<nodes_number;m++)
		{
			for(int j=0; j<Communities_num;j++) {
			random =rand.nextDouble();
			CMS_Matrix[m][j]=random;
			} 
		}
		//printMatrix(CMS_Matrix);
		int counter1=0; 
		String temp2[]= new String[2];
		try {
		File file = new File(source);
		FileReader fr = new FileReader(file);
		BufferedReader br=new BufferedReader(fr);  //creates a buffering character input stream  
		br.readLine(); br.readLine();
		
		while((line=br.readLine())!=null)  {
			temp2 = line.trim().split(" ");
			original_matrix[counter1][0] = Integer.parseInt(temp2[0]);
			original_matrix[counter1][1] = Integer.parseInt(temp2[1]);
			counter1++;
		}
		fr.close();
		}
		catch(IOException e) {}

//		System.out.println(file_length);
		//System.out.println(Communities_num);
//		for(int a=0;a<file_length;a++) {
//					System.out.print(original_matrix[a][0] + "  ");
//					System.out.println(original_matrix[a][1]);
//				}
//		
		
		//repeat on all nodes until stability
		 int z=1;
//		System.out.println("CMS_matrix before change: ");
//		printMatrix(CMS_Matrix);
//	    System.out.println("=============");
//	    for(int i=0; i<original_matrix.length; i++) {
//		 for(int j=0; j<original_matrix[0].length; j++)
//		 System.out.print(original_matrix[i][j]+"  ");
//		 System.out.println();
//	     }
//	    System.out.println("=============");

		double Sum_of_Fv[] = new double[Communities_num];  
		double Sum_Fv_Neighbor_of_u[]  = new double[Communities_num];
		double Grad_Fu[] = new double[Communities_num];
		double Sum_Fu_Exp[] = new double[Communities_num];
		double Second_Part[] = new double[Communities_num];
		double ith_vector_sum;
		boolean converged = false;
	  
		do { 
	//		while(z<3) {
		        converged=true; 
				fillZeros(Sum_of_Fv);
				for(int k=0; k<nodes_number;k++) { // calculate Sum of Fv this iteration
				 for(int l=0;l<Communities_num; l++)
					Sum_of_Fv[l] += CMS_Matrix[k][l]; 
		              }//end of calculate Sum of Fv 
				
		for(int i=0; i<nodes_number; i++) {//loop through all nodes
 		//boolean is_connected = false;
			boolean all_NAN =true;
			ith_vector_sum=0; //jump on null rows
			for(int b=0; b<Communities_num; b++) {
				ith_vector_sum += CMS_Matrix[i][b];
			}
			if(ith_vector_sum ==0)
				{
				System.out.println("There is a null vector");
				System.exit( 1);
				}

					fillZeros(Sum_Fu_Exp);
					fillZeros(Sum_Fv_Neighbor_of_u);
					fillZeros(Grad_Fu);
					fillZeros(Second_Part);
				
				double exp= 0, scalar_prod=0;
  
				for(int j =0; j<file_length; j++) {
					
					 if(original_matrix[j][1]-1 == i ) {
						//is_connected = true; 
						int sec_index=original_matrix[j][0]-1;
						scalar_prod = scalar_product(CMS_Matrix, i, sec_index);
						exp = Math.exp(scalar_prod*(-1)); 
						exp = exp/(1-exp);
							
						for(int l=0;l<Communities_num; l++) {
							Sum_Fv_Neighbor_of_u[l] += CMS_Matrix[original_matrix[j][0]-1 ] [l ];
							//Second_Part[l] = -1*(Sum_of_Fv[l] - CMS_Matrix[i][l]-Sum_Fv_Neighbor_of_u[l] ); 
							Sum_Fu_Exp[l] += (CMS_Matrix[original_matrix[j][0]-1 ] [l ] *exp);
							//Grad_Fu [l] = Sum_Fu_Exp[l] - Second_Part[l];
						}
					} 
					 else if(original_matrix[j][0]-1 == i ) {
						// is_connected = true;
    				 	scalar_prod = scalar_product(CMS_Matrix, i, original_matrix[j][1]-1);
						exp = Math.exp(scalar_prod*(-1)); 
						exp = exp/(1-exp);
						
					for(int l=0;l<Communities_num; l++) {
						Sum_Fv_Neighbor_of_u[l] += CMS_Matrix[original_matrix[j][1]-1 ] [l ];
						Sum_Fu_Exp[l] += (CMS_Matrix[original_matrix[j][1]-1 ] [l ] *exp);
					 }
				}
					
				}//end of for
//				if(!is_connected)
//					continue;
				for(int l=0;l<Communities_num; l++) {
					Grad_Fu[l] = Sum_Fu_Exp[l] - Sum_of_Fv[l] +CMS_Matrix[i][l] + Sum_Fv_Neighbor_of_u[l];
				}
//				System.out.println("Sum_Fv_not_neighbor for row: " + i);
//				for(int l=0;l<Communities_num; l++) 
//						System.out.print(sum_Fv_not_neighbor[l] + "  ");
//				System.out.println("iteration" + z +" ======");
				//double   distance=0;
			    double distance=0;
				 
				for(int l=0; l<Communities_num; l++) {
					distance+= Math.abs( CMS_Matrix[i][l]  - (CMS_Matrix[i][l] + eta*Grad_Fu[l])) ;  
					
					CMS_Matrix[i][l] = CMS_Matrix[i][l] + (eta * Grad_Fu[l]);
					if(CMS_Matrix[i][l] <0)
						CMS_Matrix[i][l] =0;
				}
				if(distance >0.01)
					converged=false;
				
		} //end of for
			
			System.out.println("+++++++++++++");
			System.out.println("Iteration number: " + z);
			printMatrix(CMS_Matrix);
			z++; 
			eta *= eta_reduction_factor; //eta must be decreased each iteration by 70%

		}while(converged == false  );
	
	
	
	
	 try {  //write results to file results.txt
	      FileWriter writer = new FileWriter("resultsCMS.txt"); 
	      int row_num = CMS_Matrix.length , col_num = CMS_Matrix[0].length ;
			for(int i=0; i<row_num; i++) {
				for(int j=0; j<col_num;j++)
	                 writer.write(CMS_Matrix[i][j]+"||\t"  );
				writer.write("\n");
			}
	      writer.close(); // close connx
	   }
	finally{}
			
	}

}
