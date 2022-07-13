import java.util.*;

public class Testing {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//Map<Set, Integer> dictionary = new HashMap<Set, Integer>();
		HashMap<Set, Integer> dictionary = new HashMap();//with HashMap we can get to the 
		Set<String> set1 = new HashSet<String> ();
		set1.add("bread");
		set1.add("water");
		set1.add("coke");
		
		Set<String> set2 = new HashSet<String>();
		set2.add("cheese");
		set2.add("water");
		
		dictionary.put(set1, 1);
	 
		int count = dictionary.containsKey(set2)? dictionary.get(set2):0;
		//System.out.println(count);
		//System.out.println(set1[0]);
		
		Set<String> comb = new HashSet<String> ();
		for(String itemi : set1) {
			comb.add(itemi);
			{
				for (String itemj : set1){
					if(itemi!=itemj) {
						
						comb.add(itemi+"-"+itemj);
				}
					}
			}
			
		}
		//System.out.println(comb);
		
		//HashMap<Set, Integer> h1 = new HashMap(); 
		Set<String > basket = new TreeSet<String>();
		basket.add("item1");
		basket.add("item2");
		basket.add("item3");
		basket.add("item4");
		basket.add("item5");
		basket.add("item6");
		Set<String > sset1 = new HashSet<String>();
		sset1.add("item1");
		Set<String > sset2 = new HashSet<String>();
		sset2.add("item2");
		Set<String > sset3 = new HashSet<String>();
		sset3.add("item3");
		Set<String > sset4 = new HashSet<String>();
		sset4.add("item4");
		Set<String > sset5 = new HashSet<String>();
		sset5.add("item5");
		
		 
		//h1.put(basket, 10);
		//for(Set s: h1.keySet())
		//System.out.println(s + ": " + h1.get(s) );
	    //basket.clear();
		//System.out.println(Arrays.asList(h1));
		
		Set<Set<String>> res = new HashSet<Set<String>>();
 
		res = generate_pairs(basket);
		HashMap<Set, Integer> A1 = new HashMap<Set, Integer>();//with HashMap we can get to the 
		A1.put(sset1, 1);
		A1.put(sset2, 1);
		A1.put(sset3, 1);
		A1.put(sset4, 1);
		A1.put(sset5, 1);
	System.out.println("A1:" +A1);
	 
		Set<String> old_basket = new TreeSet<String>();
		old_basket = basket;
		basket = new TreeSet<String>();
		basket.clear();
		//System.out.println("basket" + basket);
		System.out.println("old_basket" + old_basket);
		
		//remove from basket all items with occ < support
		Iterator<String> itr_basket = old_basket.iterator();
		while(itr_basket.hasNext()) {
			Set<String>Temp = new HashSet<String>( );
			//System.out.println("Temp" + Temp);
			Temp.add(itr_basket.next());
			//System.out.println("Temp: " + Temp);
			if(!A1.containsKey(Temp))
				itr_basket.remove(); 
			}//finish removing items < support
		
		//create all pairs and put in tempSet 
		Set<Set<String>> pairsSet = new HashSet<Set<String>>(); 
		
		
		pairsSet = generate_pairs(old_basket);
		System.out.println("pairsSet: "+pairsSet);
		Set<String> emptyset =new  HashSet<String>();
		
		Set<Set<String>> triplets_Set = new HashSet<Set<String>>(); 

		triplets_Set = generate_triplets(old_basket);
		System.out.println("Triplets: " + triplets_Set);
		
		//System.out.println(Integer.parseInt( A1.get(emptyset)) );
	} 
	//generate all combinations from basket 
	public static Set<Set<String>> generate_pairs(Set<String> s) {
		Set < Set<String>>finalSet= new HashSet< Set <String>>();
		 // in TreeSet elements will be alph sorted automactically 
		Iterator<String> it = s.iterator(); //
		 while(it.hasNext()) {
			String s1 = it.next();
			//System.out.println("n:" + n);
			//System.out.println("s1:  "  + s1);
			
			Iterator<String> it2 = s.iterator(); 
			while(it2.hasNext()) { 
				String s2 = it2.next();
				//System.out.println("s2:  " + s2);
				Set<String> setTemp = new TreeSet<String>();
				if(s1 == s2 )
					continue;
				setTemp.add(s1); 
				setTemp.add(s2);  
				finalSet.add(setTemp); 
				//System.out.println(finalSet);  
			}   
		} 
		return finalSet; 
	} 
	
	public static Set<Set<String>> generate_triplets(Set<String> s) {
		Set < Set<String>>finalSet= new HashSet< Set <String>>();
		 // in TreeSet elements will be alph sorted automactically 
		Iterator<String> it = s.iterator(); //
		 while(it.hasNext()) {
			String s1 = it.next(); 
			Iterator<String> it2 = s.iterator(); 
			while(it2.hasNext()) { 
				String s2 = it2.next();
				Iterator<String> it3 = s.iterator();
				
				while(it3.hasNext()) {
					String s3 = it3.next();
					if(s1==s2 || s2 ==s3 ||s1 ==s3)
						continue;
					Set<String> setTemp = new TreeSet<String>(); 
					setTemp.add(s1); 
					setTemp.add(s2);
					setTemp.add(s3);
					finalSet.add(setTemp); 
				} 
			}   
		} 
		return finalSet; 
	} 
	
	
	
	
	
	
	
	
	
	 
	static Set<String> generate_combinations(Set<String> s, int pass_count) {
		Set<String > results = new HashSet<String>();
	
		if (pass_count ==4) {
			Iterator<String> it = s.iterator();
			for(String item:s)
			{
				results.add(item);
				//Iterator it2 = it;
				for(String item2 :s)
					{results.add(item+"-"+item2);
						for(String item3:s)
							{results.add(item+"-"+item2+"-"+item3);
								for (String item4:s)
									results.add(item+"-"+item2+"-"+item3+"-"+item4);
							}
					}
			}
			return results;
		
			
		}
		return results;
	}
	
	static void test_combinations(Set<String> s, int Count) { 
		 Set<String > nums = new HashSet<String>();
		 Set<String > results = new HashSet<String>();
		
		 nums.add("a");
		 nums.add("b");
		 nums.add("c");
		 nums.add("d");nums.add("e");nums.add("f");
		String[] arr = {"a", "b", "c", "d" ,"e", "f", "g"};
		
		 int len = arr.length;
		 for(int i =0; i<len; i++) 
		 {
			 { results.add(arr[i]);
			 	for(int j=i+1; j<len; j++) 
			 	{
				 { if(j!=i)
						 results.add(arr[i] + arr[j]);
				 	for(int k=j+1; k<len; k++) 
				 	{ if(k!=j)
				 			results.add(arr[i]+ arr[j] + arr[k]);
				 	{ for (int l=k+1; l<len; l++) 
				 				{ if(l!=k)
				 					results.add(arr[i]+ arr[j] + arr[k]+arr[l]);
				 				{
				 					for(int m=l+1; m<len;m++) {
				 						if(m!=l)
				 							results.add(arr[i]+ arr[j] + arr[k]+arr[l]+arr[m]);
				 					}
				 					}
				 					
				 				}
				 			}
				 	}
				 }
			 }
			 }
		 
			 }
		 int count =0;
		 //System.out.println(results);
		 //System.out.println(results.size());
		 for(String s1 :results) {
			 if(s1.length()==5) {
				 System.out.print(s1+"  ");count++;
		 }}//System.out.println("");
		 //System.out.println(count);
		//return results;
	}
}

 


