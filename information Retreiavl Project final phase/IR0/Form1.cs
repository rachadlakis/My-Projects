using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Threading;



namespace IR0
{


    public partial class Form1 : Form
    {
        string[,] doubleArray;
        string[] StopListwords;
        string[] TextsNamesArray;
        Boolean isPrepared = false;


        private static string ToLongString(double input)
        {
            string str = input.ToString().ToUpper();

            // if string representation was collapsed from scientific notation, just return it:
            if (!str.Contains("E")) return str;

            bool negativeNumber = false;

            if (str[0] == '-')
            {
                str = str.Remove(0, 1);
                negativeNumber = true;
            }

            string sep = Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator;
            char decSeparator = sep.ToCharArray()[0];

            string[] exponentParts = str.Split('E');
            string[] decimalParts = exponentParts[0].Split(decSeparator);

            // fix missing decimal point:
            if (decimalParts.Length == 1) decimalParts = new string[] { exponentParts[0], "0" };

            int exponentValue = int.Parse(exponentParts[1]);

            string newNumber = decimalParts[0] + decimalParts[1];

            string result;

            if (exponentValue > 0)
            {
                result =
                    newNumber +
                    GetZeros(exponentValue - decimalParts[1].Length);
            }
            else // negative exponent
            {
                result =
                    "0" +
                    decSeparator +
                    GetZeros(exponentValue + decimalParts[0].Length) +
                    newNumber;

                result = result.TrimEnd('0');
            }

            if (negativeNumber)
                result = "-" + result;

            return result;
        }

        private static string GetZeros(int zeroCount)
        {
            if (zeroCount < 0)
                zeroCount = Math.Abs(zeroCount);

            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < zeroCount; i++) sb.Append("0");

            return sb.ToString();
        }




        public int countInArray(string s1, string[] array)
        {
            int totalNumber = 0;
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] != null)
                {
                    if (array[i].ToString() == s1.ToString())
                        totalNumber++;
                }
            }
            return totalNumber;
        }



        public string TFIDF(string TF, string DF)
        {
            double x1 = Convert.ToDouble(TF);
            double y1 = Convert.ToDouble(DF);

            double result1 = 0;
            result1 = x1 * Math.Log10(1033 / y1);
            return result1.ToString();

        }

        public string COSINE(string[] array1, string[] array2)
        {
            int lenght1 = array1.Length;
            int length2 = array2.Length;
            int length;
            if (lenght1 < length2)
                length = lenght1;
            else length = length2;

            double final = 0;
            double sq1 = 0, sq2 = 0;
            double finalUp = 0;
            for (int i = 0; i < length; i++)
            {
                double a1 = Convert.ToDouble(array1[i].ToString());
                double a2 = Convert.ToDouble(array2[i].ToString());
                double mutl = a1 * a2;
                finalUp += mutl;
                sq1 += a1 * a1;
                sq2 += a2 * a2;


            }
            final = finalUp / Math.Sqrt((sq1 * sq2));

            return final.ToString();
        }

        public Form1()
        {
            InitializeComponent();

        }

        private void button1_Click(object sender, EventArgs e)
        {
            progressBar1.Minimum = 1;
            progressBar1.Maximum = 30;
            progressBar1.Step = 1;
            progressBar1.PerformStep();


            //reaD test collection and put each paragraph in file 
            string TestCollectionPath = @"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\Test Collection\MED.ALL";
            string TestCollectionText = System.IO.File.ReadAllText(TestCollectionPath);
            string[] TestCollectionArray = TestCollectionText.Split(new string[] { ".I" }, StringSplitOptions.RemoveEmptyEntries);//each array[i] contains a text from test collection
            progressBar1.PerformStep();
            for (int i = 0; i < TestCollectionArray.Length; i++)
            {

                TestCollectionArray[i].Trim().TrimEnd().TrimStart();
            }
            progressBar1.PerformStep();
            System.IO.File.WriteAllLines(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\TestCollectionArray.txt", TestCollectionArray);

            for (int i = 1; i < TestCollectionArray.Length+1; i++)
            {
                string[] text_i = TestCollectionArray[i-1].Split(new char[] { }, StringSplitOptions.RemoveEmptyEntries);
                string text_i_name = "Text" + i.ToString();
              
                    System.IO.File.WriteAllText(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\" + text_i_name + ".txt",TestCollectionArray[i-1].ToString());
            }



            progressBar1.PerformStep();


            Porter2 porter = new Porter2();

            //get the Stop List and put in array StopListwords[]
            string filePath = @"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\StopList.txt";
            string text = System.IO.File.ReadAllText(filePath);
             StopListwords = text.Split(new char[] { ' ', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

            //get the names of all files and put in array TextNames[] and in listBox 
            string filePath1 = @"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval";
            TextsNamesArray = Directory.GetFiles(filePath1, "*.txt");

           
             string[] AllWords= { };//make a new array to put all words in all documents in it 
          
            for (int i = 0; i < TextsNamesArray.Length; i++) // loop the array containing names of all documents
            {
                string filePath2 = TextsNamesArray[i]; //get the name of the files one by one and put in filepath2
                string TextToRead = System.IO.File.ReadAllText(filePath2);

                string[] wordsOfText = TextToRead.Split(new char[] {'\'','<','>', '%','$','&','!','#','_', '�','/',':','*', '?', '+', '=', ';', '-', ' ', ',',
                        '\r','-','–','0','1','2','3','4','5','6','7','8','9', '\n', '[', ']', '(', ')', '"', '.' }, StringSplitOptions.RemoveEmptyEntries);

                //compare if words are in StopList , if not put in new array called array without stop
                String[] ArrayWithoutStop = new string[wordsOfText.Length];
                int a = 0;

                for (int j = 0; j < wordsOfText.Length; j++)
                {
                    string wordToCOmpare = wordsOfText[j].ToLower();
                    Boolean isStopWord = false;
                    for (int k = 0; k < StopListwords.Length; k++)
                    {
                        if (wordToCOmpare == StopListwords[k]) // check if word is in stop list 
                        {
                            isStopWord = true;
                            break;
                        }
                    }

                    if (isStopWord == false)
                    {
                        ArrayWithoutStop[a] = wordToCOmpare;
                        a++;
                    }
                }
                

                string result2 = "";
                result2 = Path.GetFileNameWithoutExtension(TextsNamesArray[i]);
                ArrayWithoutStop = ArrayWithoutStop.Where(c => c != null).ToArray();
                System.IO.File.WriteAllLines(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\" + result2 + ".stp", ArrayWithoutStop);

                for (int z = 0; z < ArrayWithoutStop.Length; z++)
                {
                    if (ArrayWithoutStop[z] != null)
                    {
                        String name = porter.stem(ArrayWithoutStop[z].ToLower());
                        ArrayWithoutStop[z] = name;
                    }
                }


               
                AllWords = AllWords.Concat(ArrayWithoutStop).ToArray();// try to add all arrays after stemming to the ALLWords array
                AllWords = AllWords.Where(c => c != null).ToArray();
                AllWords = AllWords.Distinct().ToArray();//remove dupliactes
                AllWords = AllWords.OrderBy(q => q).ToArray();

                string result3 = "";
                result3 = Path.GetFileNameWithoutExtension(TextsNamesArray[i]);
                ArrayWithoutStop = (ArrayWithoutStop.Where(c => c != null)).ToArray();//remove null values from ArrayWithoutStop
                System.IO.File.WriteAllLines(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\" + result2 + ".sfx", ArrayWithoutStop);
            }
           
            progressBar1.PerformStep();
            AllWords = AllWords.Distinct().ToArray();//remove duplicates from array
            AllWords = AllWords.OrderBy(i => i).ToArray();//Order in alphabetical order 
            AllWords = AllWords.Where(c => c != null).ToArray();
            int AllWordsCardinal = AllWords.Length;
           
            int fileCount = TextsNamesArray.Length;
            int totalRowsNumb = fileCount + 2;
            int totalColounNumb = AllWordsCardinal + 1;

             doubleArray = new string[totalRowsNumb, totalColounNumb];
            doubleArray[0, 0] = "Inverted File";
           
          
            for (int x = 1; x < totalColounNumb; x++)
            {
                doubleArray[0, x] = AllWords[x - 1];
            }
            progressBar1.PerformStep();
            for (int y = 1; y < totalRowsNumb - 1; y++)
            {
                string result4 = Path.GetFileNameWithoutExtension(TextsNamesArray[y - 1]);
                doubleArray[y, 0] = @"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\" + result4 + ".sfx";
            }
            progressBar1.PerformStep();
            doubleArray[totalRowsNumb - 1, 0] = "DocFrequecy";
 
            int width = doubleArray.GetLength(0);
            int height = doubleArray.GetLength(1);



            int wordRedundency = 0;
            string wordToCount = "";
           
            for (int rowNumber = 1; rowNumber < width - 1; rowNumber++)
            {
                string TextName = doubleArray[rowNumber, 0];/////

                string TextInArray = System.IO.File.ReadAllText(TextName);
                string[] wordsOfTextArray = TextInArray.Split(new char[] { ' ', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

                for (int colNumber = 1; colNumber < height; colNumber++)
                {
                    wordToCount = doubleArray[0, colNumber];
                    wordRedundency = countInArray(wordToCount, wordsOfTextArray.ToArray());
                    doubleArray[rowNumber, colNumber] = wordRedundency.ToString(); // return 0 all the time ??!!!??!!!
                }

            } progressBar1.PerformStep(); progressBar1.PerformStep();
            

            //Making DocFrequency
            int intToDocFreq = 0;
            for (int colNumber = 1; colNumber < height; colNumber++)
            {
                intToDocFreq = 0;
                for (int rowNumber = 1; rowNumber < width - 1; rowNumber++)
                {
                    if (doubleArray[rowNumber, colNumber] != "0")
                        intToDocFreq++;
                }
                doubleArray[width - 1, colNumber] = intToDocFreq.ToString();
            }

            progressBar1.PerformStep(); progressBar1.PerformStep();

         
            for (int rowNumber = 1; rowNumber < width - 1; rowNumber++)
            {
                for (int colNumber = 1; colNumber < height; colNumber++)
                {
                    doubleArray[rowNumber, colNumber] = TFIDF(doubleArray[rowNumber, colNumber], doubleArray[width - 1, colNumber]);
                }

            }

            progressBar1.PerformStep(); progressBar1.PerformStep();

            if (checkBox1.Checked == true)
            {
                System.IO.File.WriteAllLines(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\allwords.txt", AllWords);//write all disctinct words in a file called allwords

                string[] namesOfTextsinDoubleArray = new string[fileCount];

                for (int rowNumber = 1; rowNumber < totalRowsNumb - 1; rowNumber++)
                {
                    namesOfTextsinDoubleArray[rowNumber - 1] = doubleArray[rowNumber, 0];
                }

                System.IO.File.WriteAllLines(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\NamesofTextsinDoubleArray.txt", namesOfTextsinDoubleArray);
                progressBar1.PerformStep(); progressBar1.PerformStep();
               


                using (StreamWriter outfile = new StreamWriter(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\Inverted File.txt"))

                    for (int row = 0; row < width; row++)
                    {
                        string Text = "";
                        for (int col = 0; col < height; col++)
                        {
                            Text = Text + "\t" + doubleArray[row, col];

                        }
                        outfile.WriteLine(Text);
                    }
                progressBar1.PerformStep();
                //using (StreamWriter sr = new StreamWriter(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\doubleArray.txt"))
                //{
                //    foreach (var item in doubleArray)
                //    {
                //        sr.WriteLine(item);
                //    }
                //}

                
                using (StreamWriter sr = new StreamWriter(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\doubleArrayWithTFIDF.txt"))
                {
                    foreach (var item in doubleArray)
                    {
                        sr.WriteLine(item);
                    }
                }
            } progressBar1.PerformStep(); progressBar1.PerformStep();

          
            textBox1.Text = (height - 1).ToString();
            textBox2.Text = (width - 2).ToString();
            


            var autoComplete = new AutoCompleteStringCollection();
            autoComplete.AddRange(AllWords);
            textBox3.AutoCompleteCustomSource = autoComplete;

            listBox3.Visible = true;
            listBox3.DataSource = AllWords;
            label6.Visible = true;

            progressBar1.Value = 0;
            isPrepared = true;
            MessageBox.Show("The system is ready ");
        }


        private void button5_Click(object sender, EventArgs e)
        {
            progressBar1.Minimum = 1;
            progressBar1.Maximum = 15;
            progressBar1.Step = 1;
            
           

            if(isPrepared==false)
                MessageBox.Show("The Sytem is not ready yet !! ");
            else{
            if (textBox3.Text == "")
                MessageBox.Show("Enter some words to serach");
            else
            {
                progressBar1.Value = 1;
                dataGridView1.Rows.Clear();
                dataGridView1.Sort(dataGridView1.Columns[1], ListSortDirection.Descending);
                Porter2 porter3 = new Porter2();

                Porter2 porter = new Porter2();



                int width = doubleArray.GetLength(0);
                int height = doubleArray.GetLength(1);


                progressBar1.PerformStep();


                string query = textBox3.Text;//put text box in string quer
                string[] queryArray = query.Split(new char[] {'\'', '%','$','&','!','#','_', '�','/',':','*', '?', '+', '=', ';', '-', ' ', ',',
                        '\r','-','–','0','1','2','3','4','5','6','7','8','9', '\n', '[', ']', '(', ')', '"', '.' }, StringSplitOptions.RemoveEmptyEntries);//put string query in array queryArray
                string[] newQueryArray = new string[queryArray.Length]; //make new array to save words after stoplist and stemming

                progressBar1.PerformStep();
                int countinQuery = 0;
                for (int j = 0; j < queryArray.Length; j++)
                {
                    string wordToCOmpare = queryArray[j].ToLower();
                    Boolean isStopWord = false;
                    for (int k = 0; k < StopListwords.Length; k++)
                    {
                        if (wordToCOmpare == StopListwords[k]) // check if word is in stop list 
                        {
                            isStopWord = true;
                            break;
                        }
                    }

                    if (isStopWord == false)
                    {
                        newQueryArray[countinQuery] = wordToCOmpare;
                        countinQuery++;
                    }
                }
                progressBar1.PerformStep(); progressBar1.PerformStep();
                //newQueryArray = (newQueryArray.Where(c => c != null)).ToArray();
                for (int j = 0; j < newQueryArray.Length; j++)
                {
                    if (newQueryArray[j] != null)
                        newQueryArray[j] = porter3.stem(newQueryArray[j]);
                }
                progressBar1.PerformStep();
                //until here the query is stemmed and stop words are removed 
                //we duplicated the code from button1 but the doubeArray should be global !!!!!!!!!!!!!!

                string[,] newDoubleArray = new string[3, height];
                newDoubleArray[0, 0] = "AllWords";
                newDoubleArray[1, 0] = "Query";
                newDoubleArray[2, 0] = "TFIDF";
                for (int i = 1; i < height; i++)
                {
                    newDoubleArray[0, i] = doubleArray[0, i];
                    newDoubleArray[1, i] = (countInArray(newDoubleArray[0, i], newQueryArray)).ToString();
                    newDoubleArray[2, i] = TFIDF(newDoubleArray[1, i], doubleArray[width - 1, i]);
                }

                progressBar1.PerformStep(); progressBar1.PerformStep();


                string[,] CosineArray = new string[width - 1, 2];

                string[] tempquery = new string[height - 1];
                string[] tempArrayText = new string[height - 1];

                for (int a = 1; a < height; a++)
                {
                    tempquery[a - 1] = newDoubleArray[2, a];
                }
                progressBar1.PerformStep();
                //MessageBox.Show("temquery length = " + tempquery.Length + " temparraytext length = " + tempArrayText.Length);
                // MessageBox.Show("temquery first word is  = " + tempquery[0] + " temparraytext last word is  = " + tempquery[tempquery.Length - 1]);

                progressBar1.PerformStep();

                for (int i = 1; i < width - 1; i++)
                {
                    for (int j = 1; j < height; j++)
                    {

                        tempArrayText[j - 1] = doubleArray[i, j].ToString();

                    }

                    //for (int j = 1; j < height; j++)
                    //{
                    //    if (tempquery[j - 1] == null)
                    //        MessageBox.Show("null values in array after");
                    //}

                    CosineArray[i - 1, 0] = Path.GetFileNameWithoutExtension(doubleArray[i, 0]);
                    CosineArray[i - 1, 1] = COSINE(tempArrayText, tempquery);

                }

                progressBar1.PerformStep();
                if (checkBox2.Checked == true)
                {
                          //System.IO.File.WriteAllLines(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\tempquery.txt", tempquery);


                         using (StreamWriter outfile = new StreamWriter(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\Query count & TFiDF.txt"))

                              for (int row = 0; row < 3; row++)
                                  {
                                   string Text = "";
                                      for (int col = 0; col < height; col++)
                                     {
                                        Text = Text + "\t" + newDoubleArray[row, col];

                                     }
                                       outfile.WriteLine(Text);
                                  }


                         progressBar1.PerformStep();

                               // System.IO.File.WriteAllLines(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\tempArrayText.txt", tempArrayText);

                                using (StreamWriter outfile = new StreamWriter(@"C:\Users\Lakis\Desktop\Rachad\Informtion Retrieval\StopList and Results\COSINEArray.txt"))

                                    for (int row = 0; row < width - 1; row++)
                                    {
                                        string Text = "";
                                                    for (int col = 0; col < 2; col++)
                                                    {
                                                        Text = Text + "\t" + CosineArray[row, col];

                                                    }
                                        outfile.WriteLine(Text);
                                    }
                                progressBar1.PerformStep();
                     }

                dataGridView1.Visible = true;




                for (int rowIndex = 0; rowIndex < width - 2; ++rowIndex)
                {
                    var row = new DataGridViewRow();

                    for (int columnIndex = 0; columnIndex < 2; ++columnIndex)
                    {
                        if (columnIndex == 1)
                            row.Cells.Add(new DataGridViewTextBoxCell()
                            {
                                Value = double.Parse(CosineArray[rowIndex, columnIndex], System.Globalization.NumberStyles.Float)
                            });
                        else

                            row.Cells.Add(new DataGridViewTextBoxCell()
                            {
                                Value = CosineArray[rowIndex, columnIndex]
                            });
                    }

                    dataGridView1.Rows.Add(row);
                }
                progressBar1.PerformStep(); progressBar1.PerformStep();

                //Double.Parse("1.234567E-06", System.Globalization.NumberStyles.Float);

                dataGridView1.Sort(dataGridView1.Columns[1], ListSortDirection.Descending);
                listBox1.DataSource = newQueryArray;
                label4.Visible = true;
                listBox1.Visible = true;
                listBox2.DataSource = tempquery;
                label5.Visible = true;
                listBox2.Visible = true;
                progressBar1.PerformStep();
                progressBar1.Value = 1;

            }
            }
        }




        public class Porter2
        {

            string[] doubles = { "bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt" };
            string[] validLiEndings = { "c", "d", "e", "g", "h", "k", "m", "n", "r", "t" };

            private string[,] step1bReplacements =
        {
            {"eedly","ee"},
            {"ingly",""},
            {"edly",""},
            {"eed","ee"},
            {"ing",""},
            {"ed",""}
        };

            string[,] step2Replacements =
        {
            {"ization","ize"},
            {"iveness","ive"},
            {"fulness","ful"},
            {"ational","ate"},
            {"ousness","ous"},
            {"biliti","ble"},
            {"tional","tion"},
            {"lessli","less"},
            {"fulli","ful"},
            {"entli","ent"},
            {"ation","ate"},
            {"aliti","al"},
            {"iviti","ive"},
            {"ousli","ous"},
            {"alism","al"},
            {"abli","able"},
            {"anci","ance"},
            {"alli","al"},
            {"izer","ize"},
            {"enci","ence"},
            {"ator","ate"},
            {"bli","ble"},
            {"ogi","og"},
            {"li",""}
        };

            string[,] step3Replacements =
        {
            {"ational","ate"},
            {"tional","tion"},
            {"alize","al"},
            {"icate","ic"},
            {"iciti","ic"},
            {"ative",""},
            {"ical","ic"},
            {"ness",""},
            {"ful",""}
        };

            string[] step4Replacements =            
            {"ement",
            "ment",
            "able",
            "ible",
            "ance",
            "ence",
            "ate",
            "iti",
            "ion",
            "ize",
            "ive",
            "ous",
            "ant",
            "ism",
            "ent",
            "al",
            "er",
            "ic"
        };

            string[,] exceptions =
        {
        {"skis","ski"},
        {"skies","sky"},
        {"dying","die"},
        {"lying","lie"},
        {"tying","tie"},
        {"idly","idl"},
        {"gently","gentl"},
        {"ugly","ugli"},
        {"early","earli"},
        {"only","onli"},
        {"singly","singl"},
        {"sky","sky"},
        {"news","news"},
        {"howe","howe"},
        {"atlas","atlas"},
        {"cosmos","cosmos"},
        {"bias","bias"},
        {"andes","andes"}
        };

            string[] exceptions2 =
        {"inning","outing","canning","herring","earring","proceed",
            "exceed","succeed"};


            // A helper table lookup code - used for vowel lookup 
            private bool arrayContains(string[] arr, string s)
            {
                for (int i = 0; i < arr.Length; ++i)
                {
                    if (arr[i] == s) return true;
                }
                return false;
            }

            private bool isVowel(StringBuilder s, int offset)
            {
                switch (s[offset])
                {
                    case 'a':
                    case 'e':
                    case 'i':
                    case 'o':
                    case 'u':
                    case 'y':
                        return true;
                        break;
                    default:
                        return false;
                }
            }

            private bool isShortSyllable(StringBuilder s, int offset)
            {
                if ((offset == 0) && (isVowel(s, 0)) && (!isVowel(s, 1)))
                    return true;
                else
                    if (
                        ((offset > 0) && (offset < s.Length - 1)) &&
                        isVowel(s, offset) && !isVowel(s, offset + 1) &&
                        (s[offset + 1] != 'w' && s[offset + 1] != 'x' && s[offset + 1] != 'Y')
                        && !isVowel(s, offset - 1))
                        return true;
                    else
                        return false;
            }

            private bool isShortWord(StringBuilder s, int r1)
            {
                if ((r1 >= s.Length) && (isShortSyllable(s, s.Length - 2))) return true;

                return false;
            }

            private void changeY(StringBuilder sb)
            {
                if (sb[0] == 'y') sb[0] = 'Y';

                for (int i = 1; i < sb.Length; ++i)
                {
                    if ((sb[i] == 'y') && (isVowel(sb, i - 1))) sb[i] = 'Y';
                }
            }

            private void computeR1R2(StringBuilder sb, ref int r1, ref int r2)
            {
                r1 = sb.Length;
                r2 = sb.Length;

                if ((sb.Length >= 5) && (sb.ToString(0, 5) == "gener" || sb.ToString(0, 5) == "arsen")) r1 = 5;
                if ((sb.Length >= 6) && (sb.ToString(0, 6) == "commun")) r1 = 6;

                if (r1 == sb.Length) // If R1 has not been changed by exception words
                    for (int i = 1; i < sb.Length; ++i) // Compute R1 according to the algorithm
                    {
                        if ((!isVowel(sb, i)) && (isVowel(sb, i - 1)))
                        {
                            r1 = i + 1;
                            break;
                        }
                    }

                for (int i = r1 + 1; i < sb.Length; ++i)
                {
                    if ((!isVowel(sb, i)) && (isVowel(sb, i - 1)))
                    {
                        r2 = i + 1;
                        break;
                    }
                }
            }

            private void step0(StringBuilder sb)
            {

                if ((sb.Length >= 3) && (sb.ToString(sb.Length - 3, 3) == "'s'"))
                    sb.Remove(sb.Length - 3, 3);
                else
                    if ((sb.Length >= 2) && (sb.ToString(sb.Length - 2, 2) == "'s"))
                        sb.Remove(sb.Length - 2, 2);
                    else
                        if (sb[sb.Length - 1] == '\'')
                            sb.Remove(sb.Length - 1, 1);
            }

            private void step1a(StringBuilder sb)
            {

                if ((sb.Length >= 4) && sb.ToString(sb.Length - 4, 4) == "sses")
                    sb.Replace("sses", "ss", sb.Length - 4, 4);
                else
                    if ((sb.Length >= 3) && (sb.ToString(sb.Length - 3, 3) == "ied" || sb.ToString(sb.Length - 3, 3) == "ies"))
                    {
                        if (sb.Length > 4)
                            sb.Replace(sb.ToString(sb.Length - 3, 3), "i", sb.Length - 3, 3);
                        else
                            sb.Replace(sb.ToString(sb.Length - 3, 3), "ie", sb.Length - 3, 3);
                    }
                    else
                        if ((sb.Length >= 2) && (sb.ToString(sb.Length - 2, 2) == "us" || sb.ToString(sb.Length - 2, 2) == "ss"))
                            return;
                        else
                            if ((sb.Length > 0) && (sb.ToString(sb.Length - 1, 1) == "s"))
                            {
                                for (int i = 0; i < sb.Length - 2; ++i)
                                    if (isVowel(sb, i))
                                    {
                                        sb.Remove(sb.Length - 1, 1);
                                        break;
                                    }
                            }

            }

            private void step1b(StringBuilder sb, int r1)
            {
                for (int i = 0; i < 6; ++i)
                {
                    if ((sb.Length > step1bReplacements[i, 0].Length) && (sb.ToString(sb.Length - step1bReplacements[i, 0].Length, step1bReplacements[i, 0].Length) == step1bReplacements[i, 0]))
                    {
                        switch (step1bReplacements[i, 0])
                        {
                            case "eedly":
                            case "eed":
                                if (sb.Length - step1bReplacements[i, 0].Length >= r1)
                                    sb.Replace(step1bReplacements[i, 0], step1bReplacements[i, 1], sb.Length - step1bReplacements[i, 0].Length, step1bReplacements[i, 0].Length);
                                break;
                            default:
                                bool found = false;
                                for (int j = 0; j < sb.Length - step1bReplacements[i, 0].Length; ++j)
                                {
                                    if (isVowel(sb, j))
                                    {
                                        sb.Replace(step1bReplacements[i, 0], step1bReplacements[i, 1], sb.Length - step1bReplacements[i, 0].Length, step1bReplacements[i, 0].Length);
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found) return;
                                switch (sb.ToString(sb.Length - 2, 2))
                                {
                                    case "at":
                                    case "bl":
                                    case "iz":
                                        sb.Append("e");
                                        return;
                                }
                                if (arrayContains(doubles, sb.ToString(sb.Length - 2, 2)))
                                {
                                    sb.Remove(sb.Length - 1, 1);
                                    return;
                                }
                                if (isShortWord(sb, r1))
                                    sb.Append("e");
                                break;
                        }
                        return;
                    }
                }
            }

            private void step1c(StringBuilder sb)
            {
                if ((sb.Length > 0) &&
                    (sb[sb.Length - 1] == 'y' || sb[sb.Length - 1] == 'Y') &&
                    (sb.Length > 2) && (!isVowel(sb, sb.Length - 2))
                   )
                    sb[sb.Length - 1] = 'i';
            }

            private void step2(StringBuilder sb, int r1)
            {
                for (int i = 0; i < 24; ++i)
                {
                    if (
                        (sb.Length >= step2Replacements[i, 0].Length) &&
                        (sb.ToString(sb.Length - step2Replacements[i, 0].Length, step2Replacements[i, 0].Length) == step2Replacements[i, 0])
                        )
                    {
                        if (sb.Length - step2Replacements[i, 0].Length >= r1)
                        {
                            switch (step2Replacements[i, 0])
                            {
                                case "ogi":
                                    if ((sb.Length > 3) &&
                                        (sb[sb.Length - step2Replacements[i, 0].Length - 1] == 'l')
                                        )
                                        sb.Replace(step2Replacements[i, 0], step2Replacements[i, 1], sb.Length - step2Replacements[i, 0].Length, step2Replacements[i, 0].Length);
                                    return;
                                case "li":
                                    if ((sb.Length > 1) &&
                                        (arrayContains(validLiEndings, sb.ToString(sb.Length - 3, 1)))
                                        )
                                        sb.Remove(sb.Length - 2, 2);
                                    return;
                                default:
                                    sb.Replace(step2Replacements[i, 0], step2Replacements[i, 1], sb.Length - step2Replacements[i, 0].Length, step2Replacements[i, 0].Length);
                                    return;
                                    break;

                            }
                        }
                        else
                            return;
                    }
                }
            }

            private void step3(StringBuilder sb, int r1, int r2)
            {
                for (int i = 0; i < 9; ++i)
                {
                    if (
                        (sb.Length >= step3Replacements[i, 0].Length) &&
                        (sb.ToString(sb.Length - step3Replacements[i, 0].Length, step3Replacements[i, 0].Length) == step3Replacements[i, 0])
                        )
                    {
                        if (sb.Length - step3Replacements[i, 0].Length >= r1)
                        {
                            switch (step3Replacements[i, 0])
                            {
                                case "ative":
                                    if (sb.Length - step3Replacements[i, 0].Length >= r2)
                                        sb.Replace(step3Replacements[i, 0], step3Replacements[i, 1], sb.Length - step3Replacements[i, 0].Length, step3Replacements[i, 0].Length);
                                    return;
                                default:
                                    sb.Replace(step3Replacements[i, 0], step3Replacements[i, 1], sb.Length - step3Replacements[i, 0].Length, step3Replacements[i, 0].Length);
                                    return;
                            }
                        }
                        else return;
                    }
                }
            }

            private void step4(StringBuilder sb, int r2)
            {
                for (int i = 0; i < 18; ++i)
                {
                    if (
                        (sb.Length >= step4Replacements[i].Length) &&
                        (sb.ToString(sb.Length - step4Replacements[i].Length, step4Replacements[i].Length) == step4Replacements[i])                    // >=
                        )
                    {
                        if (sb.Length - step4Replacements[i].Length >= r2)
                        {
                            switch (step4Replacements[i])
                            {
                                case "ion":
                                    if (
                                        (sb.Length > 3) &&
                                        (
                                            (sb[sb.Length - step4Replacements[i].Length - 1] == 's') ||
                                            (sb[sb.Length - step4Replacements[i].Length - 1] == 't')
                                        )
                                       )
                                        sb.Remove(sb.Length - step4Replacements[i].Length, step4Replacements[i].Length);
                                    return;
                                default:
                                    sb.Remove(sb.Length - step4Replacements[i].Length, step4Replacements[i].Length);
                                    return;
                            }
                        }
                        else
                            return;
                    }
                }

            }

            private void step5(StringBuilder sb, int r1, int r2)
            {
                if (sb.Length > 0)
                    if (
                        (sb[sb.Length - 1] == 'e') &&
                        (
                            (sb.Length - 1 >= r2) ||
                            ((sb.Length - 1 >= r1) && (!isShortSyllable(sb, sb.Length - 3)))
                        )
                       )
                        sb.Remove(sb.Length - 1, 1);
                    else
                        if (
                            (sb[sb.Length - 1] == 'l') &&
                                (sb.Length - 1 >= r2) &&
                                (sb[sb.Length - 2] == 'l')
                            )
                            sb.Remove(sb.Length - 1, 1);
            }

            public string stem(string word)
            {

                if (word.Length < 3) return word; //adding to porter word!=""

                StringBuilder sb = new StringBuilder(word.ToLower());

                if (sb[0] == '\'') sb.Remove(0, 1);

                for (int i = 0; i < exceptions.Length / 2; ++i)
                    if (word == exceptions[i, 0])
                        return exceptions[i, 1];

                int r1 = 0, r2 = 0;
                changeY(sb);
                computeR1R2(sb, ref r1, ref r2);

                step0(sb);
                step1a(sb);

                for (int i = 0; i < exceptions2.Length; ++i)
                    if (sb.ToString() == exceptions2[i])
                        return exceptions2[i];

                step1b(sb, r1);
                step1c(sb);
                step2(sb, r1);
                step3(sb, r1, r2);
                step4(sb, r2);
                step5(sb, r1, r2);


                return sb.ToString().ToLower();
            }
        }

       




    }
}

    

