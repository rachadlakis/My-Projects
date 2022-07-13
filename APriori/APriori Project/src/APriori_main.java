import java.util.*;
import java.io.*; 
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map.Entry;

public class APriori_main { 
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static void main(String[] args) throws IOException {
		float support , confidence=0.6f;  
		String line = "";  
		String splitBy = ","; 
		
		Path path = Paths.get("E:\\USJ\\Data Science\\Semester "
				+ "2\\Mining Massive Datasets (Big data)\\Projects\\APriori\\APriori\\"
				+ "bread basket.csv");
		 
		//we can choose the support to be a percentage of total or a specific number 
		//support = lines_nb * 0.001f; // support now contains the minimum nb of occurences 
		support = 30f ;
		//confidence = 0.5f * support; //confidence 80% and support 1%
		
		HashMap<Set, Integer> A1 = new HashMap<Set, Integer>();//with HashMap we can get to the 
		//element directly , search and insert is O(1)
		HashMap<Set, Integer> A2 = new HashMap<Set, Integer>();
		 
		//Pass 1 : read all baskets and add elements to the HashMap A1 one by one
		try {
		BufferedReader br = new BufferedReader(new FileReader(path.toString()));  
		br.readLine();//skip first line
		while ((line = br.readLine()) != null)   //returns a Boolean value  
		{  
			String[] info = line.split(splitBy);    // use comma as separator 
			String item = info[1]; 
			Set<String> tempSet = new TreeSet<String>();
			tempSet.add(item); 
			//if already already exists, increment the counter 
			int count = A1.containsKey(tempSet)? A1.get(tempSet):0; 
			//if basket exists in Dictionary, increment
			//else add with value = 1 
			if(count != 0) //if the hashmap already contains the item add 1 to value
			{ A1.put(tempSet, count+1); }
			else { // add the item and 1 as value to hashmap
				A1.put(tempSet,1); } 
			} 
		br.close();}
		catch (IOException e)  { e.printStackTrace();  
		} // HashMap A1 now contains all items as sets as keys, with count of each one as value
		
//		for(Set s: A1.keySet())
//			System.out.println(s + ": " + A1.get(s) );
//		
		  
		//remove all element that have less that support from A1
		Set<Entry<Set,Integer>> setOfEntries = A1.entrySet();
		Iterator it = setOfEntries.iterator();
		while(it.hasNext()) {
		Entry<Set, Integer> entry =  (Entry<Set, Integer>) it.next();
		int value = entry.getValue();
		if (value < support)
			it.remove(); 
		} 
		  
		//Pass 2: add pairs to HashMap A1
		Set<String > old_basket = new TreeSet<String>();//Temporal set basket to save items
		Set<String > basket = new TreeSet<String>();
		int prev_id =-1; 
		try {
			BufferedReader br = new BufferedReader(new FileReader(path.toString()));  
			br.readLine();//skip first line (Headers)
			while ((line = br.readLine()) != null)   //returns a Boolean value  
			{  
				String[] info = line.split(splitBy);
				
				int id = Integer.parseInt(info[0]);
				String item = info[1]; 
				
				if(id == prev_id || prev_id == -1 ) {//if same basket_id only add elemnts to bakset
					//System.out.println("Hello");
					basket.add(item);
					prev_id = id;
					
					
				}
				else { //only when basket_id changes we should make the work for old_basket
					
					//remove from basket all items with occ < support
					Iterator<String> itr_basket = basket.iterator();
					while(itr_basket.hasNext()) {
						Set<String>Temp = new HashSet<String>( ); 
						Temp.add(itr_basket.next());
						if(!A1.containsKey(Temp))
							itr_basket.remove(); 
						}//finish removing items < support
					
					//create all pairs and put in tempSet 
					Set<Set<String>> pairsSet = new HashSet<Set<String>>(); 
					pairsSet = generate_pairs(basket);
					//System.out.println(pairsSet);
					//iterate over pairsSet and add them to A2
					
					Iterator<Set<String>> itr_tempPairs = pairsSet.iterator();
					while(itr_tempPairs.hasNext()) {
						Set<String> Temp2 = new TreeSet<String>();
						Temp2 = itr_tempPairs.next(); 
						int count =0; //////// 
						if(A2.get(Temp2) == null )
							A2.put(Temp2, 1);
						else { 
							count = A2.get(Temp2);
							A2.put(Temp2, count +1);} 
					}
					//for all sets in pairSet, add to A1 with val= 1 ot increment if exists in A1
					
				   //else ..... add with value =1
					//watch the last basket 
					basket.clear();
					basket.add(item); 
					prev_id = id;
				} 
				} 
			br.close();}
			catch (IOException e)  { e.printStackTrace(); } 
	 
//		//remove values from pairs less than confidence
		Set<Entry<Set,Integer>> set_OfEntries = A2.entrySet();
		Iterator it2 = set_OfEntries.iterator();
		while(it2.hasNext()) {
		Entry<Set, Integer> entry =  (Entry<Set, Integer>) it2.next();
		int value = entry.getValue();
		if (value < support)
			it2.remove(); 
		} 
		 
		//Finding associations after all pairs < confidence are eliminated
		FileWriter fw = new FileWriter("results.txt");
		fw.write("Associaiton Rules for support = " + support 
				+ " and confidecne = " + confidence+"\n\n");
		Set<Entry<Set,Integer>> set_OfEntries2 = A2.entrySet();
		Iterator it3 = set_OfEntries2.iterator();
		
		while(it3.hasNext()) {
		@SuppressWarnings("unchecked")
		Entry<Set, Integer> entry =  (Entry<Set, Integer>) it3.next();
		float value_of_pair = entry.getValue();

		Set<String> pair_Set2 = new TreeSet<String>();
		pair_Set2 = entry.getKey();
		 
		Iterator s_itr = pair_Set2.iterator();
		while(s_itr.hasNext()) {
			String s1 = (String)s_itr.next();
			Set<String> s1_Set = new TreeSet<String>();
			s1_Set.add(s1);
			float value_of_singleton = A1.get(s1_Set);
			//fw.write(pair_Set2.toString() + value_of_pair +" -> " + value_of_singleton +"\n");

  			if( (value_of_pair / value_of_singleton) > confidence) {
  				//fw.write(value_of_pair +" -> " + value_of_singleton +"\n");

  				fw.write(pair_Set2 +" -> " + s1 +"\n"); 
			}
		} 
		} 
		fw.close();
 
	}
	
	public static Set<Set<String>> generate_pairs(Set<String> s) {
		Set < Set<String>>finalSet= new HashSet< Set <String>>(); 
		Iterator<String> it = s.iterator();   
		while(it.hasNext()) {
			String s1 = it.next(); 
			Iterator<String> it2 = s.iterator(); 
			while(it2.hasNext()) { 
				String s2 = it2.next(); 
				Set<String> setTemp = new TreeSet<String>();// in TreeSet elements will be alph sorted automactically
				if(s1 == s2 )
					continue;
				setTemp.add(s1); 
				setTemp.add(s2);  
				finalSet.add(setTemp);  
			} } 
		return finalSet; 
	} 
	
}
	
