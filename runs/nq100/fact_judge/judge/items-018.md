Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 5 items.

=== ITEM id=61695981cd77e2b4 ===
Gold sentences and facts:
Sentence 0: In vertebrates , the gallbladder is a small hollow organ where bile is stored and concentrated before it is released into the small intestine . In humans , the pear - shaped gallbladder lies beneath the liver , although the structure and position of the gallbladder can vary significantly among animal species . It receives and stores bile , produced by the liver , via the common hepatic duct , and releases it via the common bile duct into the duodenum , where the bile helps in the digestion of fats .
  Fact 0.0: In vertebrates, the gallbladder is a small hollow organ.
  Fact 0.1: In vertebrates, bile is stored in the gallbladder.
  Fact 0.2: In vertebrates, bile is concentrated in the gallbladder before it is released into the small intestine.
  Fact 0.3: In humans, the gallbladder is pear-shaped.
  Fact 0.4: In humans, the gallbladder lies beneath the liver.
  Fact 0.5: The structure and position of the gallbladder can vary significantly among animal species.
  Fact 0.6: The gallbladder receives bile produced by the liver via the common hepatic duct.
  Fact 0.7: The gallbladder stores bile produced by the liver via the common hepatic duct.
  Fact 0.8: The gallbladder releases bile via the common bile duct into the duodenum.
  Fact 0.9: In the duodenum, bile helps in the digestion of fats.

Retrieved items:
<item 1>
Graph query row:
entity: Gallbladder
page: Gallbladder
passage: Gallbladder
The cystic duct unites with the common hepatic duct to become the common bile duct . At the junction of the neck of the gallbladder and the cystic duct , there
is an out - pouching of the gallbladder wall forming a mucosal fold known as `` Hartmann 's pouch '' . Microanatomy ( edit ) Micrograph of a normal gallbladder
wall . H&E stain . The gallbladder wall is composed of a number of layers . The gallbladder wall 's innermost surface is lined by a single layer of columnar
cells with a brush border of microvilli , very similar to intestinal absorptive cells . Underneath the epithelium is an underlying lamina propria , a muscular
layer , an outer perimuscular layer and serosa . Unlike elsewhere in the intestinal tract , the gallbladder does not have a muscularis mucosae , and the
muscular fibres are not arranged in distinct layers . The mucosa , the inner portion of the gallbladder wall , consists of a lining of a single layer of
columnar cells , with cells possessing small hair - like attachments called microvilli .
</item 1>

<item 2>
Graph query row:
entity: Neck of the gallbladder
page: Gallbladder
passage: Gallbladder
The cystic duct unites with the common hepatic duct to become the common bile duct . At the junction of the neck of the gallbladder and the cystic duct , there
is an out - pouching of the gallbladder wall forming a mucosal fold known as `` Hartmann 's pouch '' . Microanatomy ( edit ) Micrograph of a normal gallbladder
wall . H&E stain . The gallbladder wall is composed of a number of layers . The gallbladder wall 's innermost surface is lined by a single layer of columnar
cells with a brush border of microvilli , very similar to intestinal absorptive cells . Underneath the epithelium is an underlying lamina propria , a muscular
layer , an outer perimuscular layer and serosa . Unlike elsewhere in the intestinal tract , the gallbladder does not have a muscularis mucosae , and the
muscular fibres are not arranged in distinct layers . The mucosa , the inner portion of the gallbladder wall , consists of a lining of a single layer of
columnar cells , with cells possessing small hair - like attachments called microvilli .
</item 2>

<item 3>
Gallbladder
Function ( edit ) 1 . Bile ducts : 2 . Intrahepatic bile ducts , 3 . Left and right hepatic ducts , 4 . Common hepatic duct , 5 . Cystic duct , 6 . Common bile
duct , 7 . Ampulla of Vater , 8 . Major duodenal papilla 9 . Gallbladder , 10 -- 11 . Right and left lobes of liver . 12 . Spleen . 13 . Esophagus. 14 . Stomach
. 15 . Pancreas : 16 . Accessory pancreatic duct , 17 . Pancreatic duct . 18 . Small intestine : 19 . Duodenum , 20 . Jejunum 21 -- 22 . Right and left kidneys
. The front border of the liver has been lifted up ( brown arrow ) . The main purpose of the gallbladder is to store bile , also called gall , needed for the
digestion of fats in food .
</item 3>

<item 4>
Gallbladder
PMID 7482221 . Jump up ^ Segura - Sampedro , JJ ; Navarro - Sánchez , A ; Ashrafian , H ; Martínez - Isla , A ( February 2015 ) . `` Laparoscopic approach to
the intrahepatic gallbladder . A case report '' . Revista espanola de enfermedades digestivas : organo oficial de la Sociedad Espanola de Patologia Digestiva.
107 ( 2 ) : 122 -- 3 . PMID 25659400 . Archived from the original on March 4 , 2016 . Jump up ^ Dhulkotia , A ; Kumar , S ; Kabra , V ; Shukla , HS ( March 1 ,
2002 ) . `` Aberrant gallbladder situated beneath the left lobe of liver '' . HPB. 4 ( 1 ) : 39 -- 42 . doi : 10.1080 / 136518202753598726 .
</item 4>

<item 5>
Gallbladder
The gallbladder can be affected by gallstones , formed by material that can not be dissolved -- usually cholesterol or bilirubin , a product of haemoglobin
breakdown . These may cause significant pain , particularly in the right upper corner of the abdomen , and are often treated with removal of the gallbladder
called a cholecystectomy . Cholecystitis , inflammation of the gallbladder , has a wide range of causes , including result from the impaction of gallstones ,
infection , and autoimmune disease . Contents ( hide ) 1 Structure 1.1 Microanatomy 1.2 Variation 1.3 Development 2 Function 3 Clinical significance 3.1
Gallstones 3.2 Inflammation 3.3 Gallbladder removal 3.4 Cancer 3.5 Tests 4 Society and culture 5 Other animals 6 History 7 See also 8 References 9 External
links Structure ( edit ) The gallbladder is a hollow organ that sits in a shallow depression below the right lobe of the liver , that is grey - blue in life .
</item 5>

<item 6>
Graph query row:
entity: Gallbladder wall
page: Gallbladder
passage: Gallbladder
The cystic duct unites with the common hepatic duct to become the common bile duct . At the junction of the neck of the gallbladder and the cystic duct , there
is an out - pouching of the gallbladder wall forming a mucosal fold known as `` Hartmann 's pouch '' . Microanatomy ( edit ) Micrograph of a normal gallbladder
wall . H&E stain . The gallbladder wall is composed of a number of layers . The gallbladder wall 's innermost surface is lined by a single layer of columnar
cells with a brush border of microvilli , very similar to intestinal absorptive cells . Underneath the epithelium is an underlying lamina propria , a muscular
layer , an outer perimuscular layer and serosa . Unlike elsewhere in the intestinal tract , the gallbladder does not have a muscularis mucosae , and the
muscular fibres are not arranged in distinct layers . The mucosa , the inner portion of the gallbladder wall , consists of a lining of a single layer of
columnar cells , with cells possessing small hair - like attachments called microvilli .
</item 6>

<item 7>
Gallbladder
5 ( 1 ) : 89 -- 94 . Books editor - in - chief , Susan Standring ; section editors , Neil R. Borley ; et al. ( 2008 ) . Gray 's anatomy : the anatomical basis
of clinical practice ( 40th ed . ) . London : Churchill Livingstone . ISBN 978 - 0 - 8089 - 2371 - 8 . CS1 maint : Explicit use of et al. ( link ) CS1 maint :
Multiple names : authors list ( link ) CS1 maint : Extra text : authors list ( link ) External links ( edit ) Wikimedia Commons has media related to Gallbladder
. Diagram of Human Stomach and Gallbladder -- Human Anatomy Online dd , MyHealthScore.com . ( hide ) Anatomy of the liver ,
</item 7>

<item 8>
Gallbladder
Rarely , two or even three gallbladders may coexist , either as separate bladders draining into the cystic duct , or sharing a common branch that drains into
the cystic duct . Additionally , the gallbladder may fail to form at all . Gallbladders with two lobes separated by a septum may also exist . These
abnormalities are not likely to affect function and are generally asymptomatic . The location of the gallbladder in relation to the liver may also vary , with
documented variants including gallbladders found within , above , on the left side of , behind , and detached or suspended from the liver . Such variants are
very rare : from 1886 to 1998 , only 110 cases of left - lying liver , or less than one per year , were reported in scientific literature . An anatomical
variation can occur , known as a Phrygian cap , which is an innocuous fold in the fundus , named after its resemblance to the Phrygian cap . Development ( edit
) The gallbladder develops from an endodermal outpouching of the embryonic gut tube .
</item 8>

<item 9>
Gallbladder
doi : 10.1080 / 136518202753598726 . Jump up ^ Naganuma , S. ; Ishida , H. ; Konno , K. ; Hamashima , Y. ; Hoshino , T. ; Naganuma , H. ; Komatsuda , T. ;
Ohyama , Y. ; Yamada , N. ; Ishida , J. ; Masamune , O. ( March 6 , 2014 ) . `` Sonographic findings of anomalous position of the gallbladder '' . Abdominal
Imaging. 23 ( 1 ) : 67 -- 72 . doi : 10.1007 / s002619900287 . PMID 9437066 . Jump up ^ Meilstrup JW ; Hopper KD ; Thieme GA ( December 1991 ) . `` Imaging of
gallbladder variants '' . AJR Am J Roentgenol. 157 ( 6 ) : 1205 -- 8 . doi : 10.2214 / ajr. 157.6. 1950867 . PMID 1950867 .
</item 9>

<item 10>
Gallbladder
Gallbladder - Wikipedia Gallbladder This is the latest accepted revision , reviewed on 18 January 2018 . Jump to : navigation , search Gallbladder The
gallbladder sits beneath the liver . Details Precursor Foregut System Digestive system Artery Cystic artery Vein Cystic vein Nerve Celiac ganglia , Vagus nerve
Identifiers Latin Vesica biliaris , vesica fellea MeSH A03. 159.439 Dorlands / Elsevier 12383343 TA A05. 8.02. 001 FMA 7202 Anatomical terminology ( edit on
Wikidata ) In vertebrates , the gallbladder is a small hollow organ where bile is stored and concentrated before it is released into the small intestine . In
humans , the pear - shaped gallbladder lies beneath the liver , although the structure and position of the gallbladder can vary significantly among animal
species . It receives and stores bile , produced by the liver , via the common hepatic duct , and releases it via the common bile duct into the duodenum , where
the bile helps in the digestion of fats .
</item 10>

<item 11>
Gallbladder
In adults , the gallbladder measures approximately 7 to 10 centimetres ( 2.8 to 3.9 inches ) in length and 4 centimetres ( 1.6 in ) in diameter when fully
distended . The gallbladder has a capacity of about 50 millilitres ( 1.8 imperial fluid ounces ) . The gallbladder is shaped like a pear , with its tip opening
into the cystic duct . The gallbladder is divided into three sections : the fundus , body , and neck . The fundus is the rounded base , angled so that it faces
the abdominal wall . The body lies in a depression in the surface of the lower liver . The neck tapers and is continuous with the cystic duct , part of the
biliary tree . The gallbladder fossa , against which the fundus and body of the gallbladder lie , is found beneath the junction of hepatic segments IVB and V .
The cystic duct unites with the common hepatic duct to become the common bile duct .
</item 11>

<item 12>
Gallbladder
This sits on a thin layer of connective tissue , the lamina propria . The mucosa is curved and collected into tiny outpouchings called rugae . A muscular layer
sits beneath the mucosa . This is formed by smooth muscle , with fibres that lie in longitudinal , oblique and transverse directions , and are not arranged in
separate layers . The muscle fibres here contract to expel bile from the gallbladder . A distinctive feature of the gallbladder is the presence of Rokitansky --
Aschoff sinuses , deep outpouchings of the mucosa that can extend through the muscular layer . The muscular layer is surrounded by a layer of connective and fat
tissue . The outer layer of the fundus of gallbladder , and the surfaces not in contact with the liver , are covered by a thick serosa , which is exposed to the
peritoneum . The serosa contains blood vessels and lymphatics . The surfaces in contact with the liver are covered in connective tissue . Variation ( edit ) The
gallbladder varies in size , shape , and position between different people .
</item 12>

<item 13>
Gallbladder
( hide ) Anatomy of the liver , pancreas and biliary tree Liver Bare area Cantlie line Ligamentum venosum Porta hepatis Round ligament Lobes of liver Caudate
Quadrate Fibrous capsule of Glisson Perisinusoidal space Liver sinusoid Periportal space Portal triad Lobules of liver Liver segment Microanatomy Hepatocyte
Hepatic stellate cell Kupffer cell Biliary tract Bile ducts Intrahepatic bile ducts Bile canaliculus Canals of Hering Interlobular Left hepatic duct Right
hepatic duct Common hepatic duct Gallbladder Cystic duct Common bile duct Ampulla of Vater Sphincter of Oddi Cells Cholecystocyte Pancreas Gross Tail Body Neck
Head ( Uncinate process ) Ducts main accessory Microanatomy Pancreatic islets Centroacinar cell Stellate cell GND : 4071544 - 9 NDL : 00572670 Retrieved from ``
https://en.wikipedia.org/w/index.
</item 13>
=== END ITEM id=61695981cd77e2b4 ===

=== ITEM id=96eaf03b753467e4 ===
Gold sentences and facts:
Sentence 0: The Director of the United States Mint is the head of the United States Mint . The position is currently vacant . Since January 20 , 2017 , the senior career official at the Mint has been Acting Principal Deputy Director David Motl .
  Fact 0.0: The Director of the United States Mint is the head of the United States Mint.
  Fact 0.1: The position of Director of the United States Mint is currently vacant.
  Fact 0.2: Since January 20, 2017, the senior career official at the Mint has been Acting Principal Deputy Director David Motl.
Sentence 1: Name Assumed Office Left Office President Appointed By David Rittenhouse April 1792 June 1795 George Washington Henry William de Saussure July 1795 October 1795 Elias Boudinot October 1795 July 1805 Robert Patterson January 1806 July 1824 Thomas Jefferson Samuel Moore July 1824 July 1835 James Monroe Robert M. Patterson May 1835 July 1851 Andrew Jackson George Nicholas Eckert July 1851 April 1853 Millard Fillmore Thomas M. Pettit April 1853 May 1853 Franklin Pierce James Ross Snowden June 1853 May 1861 James Pollock May 1861 September 1866 Abraham Lincoln William Millward October 1866 April 1867 Andrew Johnson Henry Linderman April 1867 May 1869 James Pollock May 1869 March 1873 Ulysses S. Grant Henry Linderman April 1873 December 1878 Horatio C. Burchard February 1879 June 1885 Rutherford B. Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O. Leech October 1889 May 1893 Benjamin Harrison Robert E. Preston November 1893 February 1898 Grover Cleveland George E. Roberts February 1898 July 1907 William McKinley Frank A. Leach September 1907 August 1909 Theodore Roosevelt A. Piatt Andrew November 1909 June 1910 William Howard Taft George E. Roberts July 1910 November 1914 Robert W. Woolley March 1915 July 1916 Woodrow Wilson F.H. von Engelken September 1916 March 1917 Raymond T. Baker March 1917 March 1922 Frank Edgar Scobey March 1922 September 1923 Warren G. Harding Robert J. Grant November 1923 May 1933 Calvin Coolidge Nellie Tayloe Ross May 1933 April 1953 Franklin D. Roosevelt William H. Brett July 1954 January 1961 Dwight D. Eisenhower Eva Adams October 1961 August 1969 John F. Kennedy Mary Brooks September 1969 February 1977 Richard Nixon Stella Hackel Sims November 1977 April 1981 Jimmy Carter Donna Pope July 1981 August 1991 Ronald Reagan David J. Ryder September 1992 November 1993 George H.W. Bush Philip N. Diehl June 1994 March 2000 Bill Clinton Jay W. Johnson May 2000 August 2001 Henrietta H. Fore August 2001 August 2005 George W. Bush Edmund C. Moy September 2006 January 2011 David J. Ryder April 2018 Incumbent Donald Trump
  Fact 1.0: David Rittenhouse assumed office in April 1792 and left office in June 1795; George Washington is listed after this entry.
  Fact 1.1: Henry William de Saussure assumed office in July 1795 and left office in October 1795; Elias Boudinot is listed after this entry.
  Fact 1.2: Elias Boudinot assumed office in October 1795 and left office in July 1805; Robert Patterson is listed after this entry.
  Fact 1.3: Robert Patterson assumed office in January 1806 and left office in July 1824; Thomas Jefferson is listed after this entry.
  Fact 1.4: Samuel Moore assumed office in July 1824 and left office in July 1835; James Monroe is listed after this entry.
  Fact 1.5: Robert M. Patterson assumed office in May 1835 and left office in July 1851; Andrew Jackson is listed after this entry.
  Fact 1.6: George Nicholas Eckert assumed office in July 1851 and left office in April 1853; Millard Fillmore is listed after this entry.
  Fact 1.7: Thomas M. Pettit assumed office in April 1853 and left office in May 1853; Franklin Pierce is listed after this entry.
  Fact 1.8: James Ross Snowden assumed office in June 1853 and left office in May 1861; James Pollock is listed after this entry.
  Fact 1.9: James Pollock assumed office in May 1861 and left office in September 1866; Abraham Lincoln is listed after this entry.
  Fact 1.10: William Millward assumed office in October 1866 and left office in April 1867; Andrew Johnson is listed after this entry.
  Fact 1.11: Henry Linderman assumed office in April 1867 and left office in May 1869; James Pollock is listed after this entry.
  Fact 1.12: James Pollock assumed office in May 1869 and left office in March 1873; Ulysses S. Grant is listed after this entry.
  Fact 1.13: Henry Linderman assumed office in April 1873 and left office in December 1878; Horatio C. Burchard is listed after this entry.
  Fact 1.14: Horatio C. Burchard assumed office in February 1879 and left office in June 1885; Rutherford B. Hayes is listed after this entry.
  Fact 1.15: James P. Kimball assumed office in July 1885 and left office in October 1889; Grover Cleveland is listed after this entry.
  Fact 1.16: Edward O. Leech assumed office in October 1889 and left office in May 1893; Benjamin Harrison is listed after this entry.
  Fact 1.17: Robert E. Preston assumed office in November 1893 and left office in February 1898; Grover Cleveland is listed after this entry.
  Fact 1.18: George E. Roberts assumed office in February 1898 and left office in July 1907; William McKinley is listed after this entry.
  Fact 1.19: Frank A. Leach assumed office in September 1907 and left office in August 1909; Theodore Roosevelt is listed after this entry.
  Fact 1.20: A. Piatt Andrew assumed office in November 1909 and left office in June 1910; William Howard Taft is listed after this entry.
  Fact 1.21: George E. Roberts assumed office in July 1910 and left office in November 1914; Robert W. Woolley is listed after this entry.
  Fact 1.22: Robert W. Woolley assumed office in March 1915 and left office in July 1916; Woodrow Wilson is listed after this entry.
  Fact 1.23: F.H. von Engelken assumed office in September 1916 and left office in March 1917; Raymond T. Baker is listed after this entry.
  Fact 1.24: Raymond T. Baker assumed office in March 1917 and left office in March 1922; Frank Edgar Scobey is listed after this entry.
  Fact 1.25: Frank Edgar Scobey assumed office in March 1922 and left office in September 1923; Warren G. Harding is listed after this entry.
  Fact 1.26: Robert J. Grant assumed office in November 1923 and left office in May 1933; Calvin Coolidge is listed after this entry.
  Fact 1.27: Nellie Tayloe Ross assumed office in May 1933 and left office in April 1953; Franklin D. Roosevelt is listed after this entry.
  Fact 1.28: William H. Brett assumed office in July 1954 and left office in January 1961; Dwight D. Eisenhower is listed after this entry.
  Fact 1.29: Eva Adams assumed office in October 1961 and left office in August 1969; John F. Kennedy is listed after this entry.
  Fact 1.30: Mary Brooks assumed office in September 1969 and left office in February 1977; Richard Nixon is listed after this entry.
  Fact 1.31: Stella Hackel Sims assumed office in November 1977 and left office in April 1981; Jimmy Carter is listed after this entry.
  Fact 1.32: Donna Pope assumed office in July 1981 and left office in August 1991; Ronald Reagan is listed after this entry.
  Fact 1.33: David J. Ryder assumed office in September 1992 and left office in November 1993; George H.W. Bush is listed after this entry.
  Fact 1.34: Philip N. Diehl assumed office in June 1994 and left office in March 2000; Bill Clinton is listed after this entry.
  Fact 1.35: Jay W. Johnson assumed office in May 2000 and left office in August 2001; Henrietta H. Fore is listed after this entry.
  Fact 1.36: Henrietta H. Fore assumed office in August 2001 and left office in August 2005; George W. Bush is listed after this entry.
  Fact 1.37: Edmund C. Moy assumed office in September 2006 and left office in January 2011; David J. Ryder is listed after this entry.
  Fact 1.38: David J. Ryder assumed office in April 2018 and is listed as incumbent; Donald Trump is listed after this entry.

Retrieved items:
<item 1>
Graph query row:
person: Matthew Rhett Jeppson
relationship: PARTICIPATED_IN
connected: Nomination of Matthew Rhett Jeppson as Director of the United States Mint
</item 1>

<item 2>
Graph query row:
person: Matthew Rhett Jeppson
relationship: RELATED_TO
connected: Principal Deputy Director
</item 2>

<item 3>
Graph query row:
person: David Motl
relationship: WORKS_FOR
connected: United States Mint
</item 3>

<item 4>
Graph query row:
person: Matthew Rhett Jeppson
relationship: PARTICIPATED_IN
connected: Matthew Rhett Jeppson stepped down as Principal Deputy Director
</item 4>

<item 5>
Director of the United States Mint
Woolley March 1915 July 1916 Woodrow Wilson F.H. von Engelken September 1916 March 1917 Raymond T. Baker March 1917 March 1922 Frank Edgar Scobey March 1922
September 1923 Warren G. Harding Robert J. Grant November 1923 May 1933 Calvin Coolidge Nellie Tayloe Ross May 1933 April 1953 Franklin D. Roosevelt William H.
Brett July 1954 January 1961 Dwight D. Eisenhower Eva Adams October 1961 August 1969 John F. Kennedy Mary Brooks September 1969 February 1977 Richard Nixon
Stella Hackel Sims November 1977 April 1981 Jimmy Carter Donna Pope July 1981 August 1991 Ronald Reagan David J. Ryder September 1992 November 1993 George H.W.
</item 5>

<item 6>
Director of the United States Mint
US Mint . Retrieved 13 April 2017 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Director_of_the_United_States_Mint&oldid=836791366 ''
Categories : Directors of the United States Mint United States Department of the Treasury 1792 establishments in the United States Talk About Wikipedia Add
links This page was last edited on 16 April 2018 , at 21 : 26 . About Wikipedia
</item 6>

<item 7>
Graph query row:
person: David Motl
relationship: RELATED_TO
connected: Acting Principal Deputy Director
</item 7>

<item 8>
Graph query row:
person: Edmund C. Moy
relationship: PARTICIPATED_IN
connected: Resignation of Edmund C. Moy
</item 8>

<item 9>
Graph query row:
person: David Rittenhouse
relationship: WORKS_FOR
connected: United States Mint
</item 9>

<item 10>
Graph query row:
person: Edmund C. Moy
relationship: WORKS_FOR
connected: United States Mint
</item 10>

<item 11>
Director of the United States Mint
Bush Philip N. Diehl June 1994 March 2000 Bill Clinton Jay W. Johnson May 2000 August 2001 Henrietta H. Fore August 2001 August 2005 George W. Bush Edmund C.
Moy September 2006 January 2011 David J. Ryder April 2018 Incumbent Donald Trump References ( edit ) Jump up ^ About the Mint at usmint.gov ( retrieved 20 April
2015 ) . Jump up ^ Gilkes , Paul ( 23 January 2017 ) . `` Rhett Jeppson leaves U.S. Mint as principal deputy director '' . coinworld.com . Retrieved 13 April
2017 . Jump up ^ `` Director 's Office '' . US Mint . Retrieved 13 April 2017 .
</item 11>

<item 12>
Director of the United States Mint
Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O. Leech October 1889 May 1893 Benjamin Harrison Robert E. Preston November 1893 February
1898 Grover Cleveland George E. Roberts February 1898 July 1907 William McKinley Frank A. Leach September 1907 August 1909 Theodore Roosevelt A. Piatt Andrew
November 1909 June 1910 William Howard Taft George E. Roberts July 1910 November 1914 Robert W. Woolley March 1915 July 1916 Woodrow Wilson F.H.
</item 12>

<item 13>
Director of the United States Mint
List of Directors of the United States Mint ( edit ) Name Assumed Office Left Office President Appointed By David Rittenhouse April 1792 June 1795 George
Washington Henry William de Saussure July 1795 October 1795 Elias Boudinot October 1795 July 1805 Robert Patterson January 1806 July 1824 Thomas Jefferson
Samuel Moore July 1824 July 1835 James Monroe Robert M. Patterson May 1835 July 1851 Andrew Jackson George Nicholas Eckert July 1851 April 1853 Millard Fillmore
Thomas M. Pettit April 1853 May 1853 Franklin Pierce James Ross Snowden June 1853 May 1861 James Pollock May 1861 September 1866 Abraham Lincoln William
Millward October 1866 April 1867 Andrew Johnson Henry Linderman April 1867 May 1869 James Pollock May 1869 March 1873 Ulysses S. Grant Henry Linderman April
1873 December 1878 Horatio C. Burchard February 1879 June 1885 Rutherford B. Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O.
</item 13>

<item 14>
Graph query row:
person: Matthew Rhett Jeppson
relationship: WORKS_FOR
connected: United States Mint
</item 14>

<item 15>
Director of the United States Mint
Director of the United States Mint - wikipedia Director of the United States Mint Jump to : navigation , search Seal of the United States Mint . The Director of
the United States Mint is the head of the United States Mint . The position is currently vacant . Since January 20 , 2017 , the senior career official at the
Mint has been Acting Principal Deputy Director David Motl . The office of Director has existed since the creation of the Mint by the Coinage Act of 1792 .
Initially appointed serving at the pleasure of the President of the United States , the Coinage Act of 1873 specified a five - year term for Directors . The
Director operates with general directions provided by the United States Secretary of the Treasury . Since the resignation of Edmund C. Moy in 2011 the mint has
been without an official director . Initially Matthew Rhett Jeppson was nominated by US President Barack Obama to become the mint 's 39th director and was given
the temporary title of principal deputy director , however the nomination was never confirmed by the Senate . Jeppson stepped down as principal deputy director
in January 2017 , being replaced by Acting Principal Deputy Director David Motl .
</item 15>

<item 16>
Graph query row:
person: David Motl
relationship: PARTICIPATED_IN
connected: Matthew Rhett Jeppson stepped down as Principal Deputy Director
</item 16>
=== END ITEM id=96eaf03b753467e4 ===

=== ITEM id=68d48186ed2c3326 ===
Gold sentences and facts:
Sentence 0: Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st Emmanuel Amunike Sporting CP 48 2nd George Weah Paris Saint - Germain 44 Rashidi Yekini Olympiacos 1995 1st George Weah Milan -- 2nd Emmanuel Amunike Sporting CP -- 3rd Daniel Amokachi Everton -- 1st Nwankwo Kanu Internazionale -- 2nd George Weah Milan -- 3rd Daniel Amokachi Beşiktaş -- 1st Victor Ikpeba Monaco 56 2nd Japhet N'Doram Monaco 40 3rd Taribo West Internazionale 35 1998 1st Mustapha Hadji Deportivo La Coruña 76 2nd Jay - Jay Okocha Paris Saint - Germain 74 3rd Sunday Oliseh Ajax 58 1999 1st Nwankwo Kanu Arsenal 46 2nd Samuel Kuffour Bayern Munich 44 3rd Ibrahima Bakayoko Marseille 42 2000 1st Patrick M'Boma Parma 123 2nd Lauren Mallorca 36 3rd Samuel Eto'o Mallorca 29 1st El Hadji Diouf Lens 93 2nd Samuel Kuffour Bayern Munich 66 3rd Samuel Eto'o Mallorca 34 2002 1st El Hadji Diouf Liverpool 93 2nd Papa Bouba Diop Lens 46 3rd Mido Ajax 42 2003 1st Samuel Eto'o Mallorca -- 2nd Didier Drogba Marseille -- 3rd Jay - Jay Okocha Bolton Wanderers -- 1st Samuel Eto'o Barcelona 116 2nd Didier Drogba Chelsea 90 3rd Jay - Jay Okocha Bolton Wanderers 68 2005 1st Samuel Eto'o Barcelona 108 2nd Didier Drogba Chelsea 106 3rd Michael Essien Chelsea 50 2006 1st Didier Drogba Chelsea 79 2nd Samuel Eto'o Barcelona 76 3rd Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74 2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea 43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311
  Fact 0.0: In 1992, Abedi Pele of Marseille was ranked 1st with no points listed.
  Fact 0.1: In 1993, Rashidi Yekini of Vitória de Setúbal was ranked 1st with no points listed.
  Fact 0.2: In 1994, Emmanuel Amunike of Sporting CP was ranked 1st with 48 points.
  Fact 0.3: In 1994, George Weah of Paris Saint-Germain was ranked 2nd with 44 points.
  Fact 0.4: In 1994, Rashidi Yekini of Olympiacos is listed with no rank or points.
  Fact 0.5: In 1995, George Weah of Milan was ranked 1st with no points listed.
  Fact 0.6: In 1995, Emmanuel Amunike of Sporting CP was ranked 2nd with no points listed.
  Fact 0.7: In 1995, Daniel Amokachi of Everton was ranked 3rd with no points listed.
  Fact 0.8: An entry with no year shown has Nwankwo Kanu of Internazionale ranked 1st with no points listed.
  Fact 0.9: The entry with no year shown has George Weah of Milan ranked 2nd with no points listed.
  Fact 0.10: The entry with no year shown has Daniel Amokachi of Beşiktaş ranked 3rd with no points listed.
  Fact 0.11: An entry with no year shown has Victor Ikpeba of Monaco ranked 1st with 56 points.
  Fact 0.12: The entry with no year shown has Japhet N'Doram of Monaco ranked 2nd with 40 points.
  Fact 0.13: The entry with no year shown has Taribo West of Internazionale ranked 3rd with 35 points.
  Fact 0.14: In 1998, Mustapha Hadji of Deportivo La Coruña was ranked 1st with 76 points.
  Fact 0.15: In 1998, Jay-Jay Okocha of Paris Saint-Germain was ranked 2nd with 74 points.
  Fact 0.16: In 1998, Sunday Oliseh of Ajax was ranked 3rd with 58 points.
  Fact 0.17: In 1999, Nwankwo Kanu of Arsenal was ranked 1st with 46 points.
  Fact 0.18: In 1999, Samuel Kuffour of Bayern Munich was ranked 2nd with 44 points.
  Fact 0.19: In 1999, Ibrahima Bakayoko of Marseille was ranked 3rd with 42 points.
  Fact 0.20: In 2000, Patrick M'Boma of Parma was ranked 1st with 123 points.
  Fact 0.21: In 2000, Lauren of Mallorca was ranked 2nd with 36 points.
  Fact 0.22: In 2000, Samuel Eto'o of Mallorca was ranked 3rd with 29 points.
  Fact 0.23: An entry with no year shown has El Hadji Diouf of Lens ranked 1st with 93 points.
  Fact 0.24: The entry with no year shown has Samuel Kuffour of Bayern Munich ranked 2nd with 66 points.
  Fact 0.25: The entry with no year shown has Samuel Eto'o of Mallorca ranked 3rd with 34 points.
  Fact 0.26: In 2002, El Hadji Diouf of Liverpool was ranked 1st with 93 points.
  Fact 0.27: In 2002, Papa Bouba Diop of Lens was ranked 2nd with 46 points.
  Fact 0.28: In 2002, Mido of Ajax was ranked 3rd with 42 points.
  Fact 0.29: In 2003, Samuel Eto'o of Mallorca was ranked 1st with no points listed.
  Fact 0.30: In 2003, Didier Drogba of Marseille was ranked 2nd with no points listed.
  Fact 0.31: In 2003, Jay-Jay Okocha of Bolton Wanderers was ranked 3rd with no points listed.
  Fact 0.32: An entry with no year shown has Samuel Eto'o of Barcelona ranked 1st with 116 points.
  Fact 0.33: The entry with no year shown has Didier Drogba of Chelsea ranked 2nd with 90 points.
  Fact 0.34: The entry with no year shown has Jay-Jay Okocha of Bolton Wanderers ranked 3rd with 68 points.
  Fact 0.35: In 2005, Samuel Eto'o of Barcelona was ranked 1st with 108 points.
  Fact 0.36: In 2005, Didier Drogba of Chelsea was ranked 2nd with 106 points.
  Fact 0.37: In 2005, Michael Essien of Chelsea was ranked 3rd with 50 points.
  Fact 0.38: In 2006, Didier Drogba of Chelsea was ranked 1st with 79 points.
  Fact 0.39: In 2006, Samuel Eto'o of Barcelona was ranked 2nd with 76 points.
  Fact 0.40: In 2006, Michael Essien of Chelsea was ranked 3rd with 36 points.
  Fact 0.41: In 2007, Frédéric Kanouté of Sevilla was ranked 1st with no points listed.
  Fact 0.42: In 2007, Michael Essien of Chelsea was ranked 2nd with no points listed.
  Fact 0.43: In 2007, Didier Drogba of Chelsea was ranked 3rd with no points listed.
  Fact 0.44: In 2008, Emmanuel Adebayor of Arsenal was ranked 1st with 74 points.
  Fact 0.45: In 2008, Mohamed Aboutrika of Al-Ahly was ranked 2nd with 53 points.
  Fact 0.46: In 2008, Michael Essien of Chelsea was ranked 3rd with 36 points.
  Fact 0.47: In 2009, Didier Drogba of Chelsea was ranked 1st with 92 points.
  Fact 0.48: In 2009, Samuel Eto'o of Internazionale was ranked 2nd with 69 points.
  Fact 0.49: In 2009, Michael Essien of Chelsea was ranked 3rd with 43 points.
  Fact 0.50: An entry with no year shown has Samuel Eto'o of Internazionale ranked 1st with no points listed.
  Fact 0.51: The entry with no year shown has Asamoah Gyan of Sunderland ranked 2nd with no points listed.
  Fact 0.52: The entry with no year shown has Didier Drogba of Chelsea ranked 3rd with no points listed.
  Fact 0.53: In 2011, Yaya Touré of Manchester City was ranked 1st with no points listed.
  Fact 0.54: In 2011, Seydou Keita of Barcelona was ranked 2nd with no points listed.
  Fact 0.55: In 2011, André Ayew of Marseille was ranked 3rd with no points listed.
  Fact 0.56: In 2012, Yaya Touré of Manchester City was ranked 1st with no points listed.
  Fact 0.57: In 2012, Didier Drogba of Shanghai Shenhua was ranked 2nd with no points listed.
  Fact 0.58: In 2012, Alex Song of Barcelona was ranked 3rd with no points listed.
  Fact 0.59: In 2013, Yaya Touré of Manchester City was ranked 1st with 373 points.
  Fact 0.60: In 2013, John Obi Mikel of Chelsea was ranked 2nd with 265 points.
  Fact 0.61: In 2013, Didier Drogba of Galatasaray was ranked 3rd with 236 points.
  Fact 0.62: In 2014, Yaya Touré of Manchester City was ranked 1st with 175 points.
  Fact 0.63: In 2014, Pierre-Emerick Aubameyang of Borussia Dortmund was ranked 2nd with 120 points.
  Fact 0.64: In 2014, Vincent Enyeama of Lille was ranked 3rd with 105 points.
  Fact 0.65: In 2015, Pierre-Emerick Aubameyang of Borussia Dortmund was ranked 1st with 143 points.
  Fact 0.66: In 2015, Yaya Touré of Manchester City was ranked 2nd with 136 points.
  Fact 0.67: In 2015, André Ayew of Swansea City was ranked 3rd with 112 points.
  Fact 0.68: In 2016, Riyad Mahrez of Leicester City was ranked 1st with 361 points.
  Fact 0.69: In 2016, Pierre-Emerick Aubameyang of Borussia Dortmund was ranked 2nd with 313 points.
  Fact 0.70: In 2016, Sadio Mané of Liverpool was ranked 3rd with 186 points.
  Fact 0.71: In 2017, Mohamed Salah of Liverpool was ranked 1st with 625 points.
  Fact 0.72: In 2017, Sadio Mané of Liverpool was ranked 2nd with 507 points.
  Fact 0.73: In 2017, Pierre-Emerick Aubameyang of Borussia Dortmund was ranked 3rd with 311 points.

Retrieved items:
<item 1>
African Footballer of the Year
Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74
2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea
43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita
Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya
Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014
</item 1>

<item 2>
African Footballer of the Year
African Footballer of the Year - wikipedia African Footballer of the Year Jump to : navigation , search The African Footballer of the Year award , presented to
the best African footballer each year , has been conferred by the Confederation of African Football ( CAF ) since 1992 . An earlier African Footballer of the
Year Golden Ball award was given out between 1970 and 1994 by France Football magazine . The changes resulted in parallel Golden Ball awards given out to Abedi
Pele and George Weah in 1993 and 1994 by the magazine although the CAF sponsored awards for those years were won respectively by Rashidi Yekini and Emmanuel
Amuneke , as well as two awards given to Abedi Pelé in 1992 . France Football discontinued the election from 1995 after the European Ballon d'Or -- also awarded
by the magazine -- had been opened to all players in the European leagues . On 2 February 2008 , Sevilla 's Frédéric Kanouté was named the 2007 African
Footballer of the Year .
</item 2>

<item 3>
African Footballer of the Year
Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd Roger Milla Tonnerre Yaoundé 24 Mamadou Aliou Kéïta Hafia 1976 1st Roger Milla Tonnerre Yaoundé 33 2nd Papa Camara
Hafia 32 3rd Ali Bencheikh MC Alger 27 1977 1st Tarak Dhiab Espérance Tunis 45 2nd Papa Camara Hafia 33 3rd Segun Odegbami Shooting Stars 29 1978 1st Karim
Abdul Razak Asante Kotoko 58 2nd Ali Bencheikh MC Alger 33 3rd Thomas N'Kono Canon Yaoundé 29 1979 1st Thomas N'Kono Canon Yaoundé 55 2nd Adolf Armah Hearts of
Oak 23 3rd Kerfalla Bangoura Horoya 15 1980 1st Jean Manga - Onguéné Canon Yaoundé 64 2nd Segun Odegbami Shooting Stars 41 3rd Théophile Abega Canon Yaoundé
</item 3>

<item 4>
African Footballer of the Year
Cannes 52 1992 1st Abedi Pele Marseille 198 2nd George Weah Paris Saint - Germain 161 3rd Tony Yeboah Eintracht Frankfurt 64 1993 1st Abedi Pele Marseille 119
2nd Tony Yeboah Eintracht Frankfurt 117 3rd Rashidi Yekini Vitória de Setúbal 104 1994 1st George Weah Paris Saint - Germain 148 2nd Emmanuel Amunike Sporting
CP 133 3rd Daniel Amokachi Everton 99 CAF award ( 1992 -- present ) ( edit ) Samuel Eto'o won the award four times . Yaya Touré won the award four consecutive
times from 2011 to 2014 . Pierre - Emerick Aubameyang was the first player from Gabon to win the award in 2015 . Mohamed Salah is the latest player to win the
award in 2017 . Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st
</item 4>

<item 5>
African Footballer of the Year
Keïta 1971 : Sunday 1972 : Souleymane 1973 : Bwanga 1974 : Moukila 1975 : Faras 1976 : Milla 1977 : Dhiab 1978 : Abdul Razak 1979 : N'Kono 1980 : Manga -
Onguéné 1981 : Belloumi 1982 : N'Kono 1983 : El Khatib 1984 : Abega 1985 : Timoumi 1986 : Zaki 1987 : Madjer 1988 : Bwalya 1989 : Weah 1990 : Milla 1991 : Abedi
Pele 1992 : Abedi Pele 1993 : Abedi Pele 1994 : Weah CAF award 1992 : Abedi Pele 1993 : Yekini 1994 : Amunike 1995 : Weah 1996 : Kanu 1997 : Ikpeba 1998 : Hadji
1999 : Kanu 2000 : M'Boma 2001 : Diouf 2002 :
</item 5>

<item 6>
African Footballer of the Year
It had already been replaced by an official award given out by the Confederation of African Football since 1992 . Year Rank Player Team Points 1970 1st Salif
Keïta Saint - Étienne 54 2nd Laurent Pokou ASEC Abidjan 28 Ali Abo Greisha Ismaily 1971 1st Ibrahim Sunday Asante Kotoko 29 2nd Robert Mensah Asante Kotoko 15
3rd François Ndoumbé Canon Yaoundé 13 1972 1st Chérif Souleymane Hafia 21 2nd Bwanga Tshimen TP Mazembe 16 3rd Petit Sory Hafia 14 1973 1st Bwanga Tshimen TP
Mazembe 49 2nd Kazadi Mwamba TP Mazembe 44 3rd Laurent Pokou ASEC Abidjan 41 1st Paul Moukila CARA Brazzaville 57 2nd Lobilo Boba Vita Club 32 3rd Hassan
Shehata Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd
</item 6>

<item 7>
African Footballer of the Year
Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC Mulhouse -- -- Hearts of Oak -- -- FC Sion -- -- Aydınspor -- -- JS Kabylie -- --
Bolton Wanderers -- -- Everton FC -- -- Ajax Amsterdam -- -- Ismaily SC -- -- Zamalek SC -- AS Cannes -- -- Stade Lavallois -- -- Lille OSC -- -- Stade Rennais
-- -- Olympiacos -- -- Horoya AC -- -- PSV Eindhoven -- -- OC Agaza -- -- Beşiktaş -- -- Galatasaray -- -- Swansea City -- -- See also ( edit ) BBC African
Footballer of the Year African Women Footballer of the Year FIFA World Player of the Year CAF Awards References ( edit ) Jump up ^ `` African Player of the Year
1993 '' . Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year 1994 '' . Rsssf.com. 2001 - 01 - 05 .
</item 7>

<item 8>
African Footballer of the Year
Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama
Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez
Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané
Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311 Wins By Country ( edit ) Nation Winners Runners - up Third places Cameroon 11 8 11 Ivory
Coast 8 6 7 Ghana 6 7 7 Nigeria 5 6 7 Morocco 0 Algeria 5 Liberia -- Egypt Mali -- Senegal Guinea DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- --
Chad -- -- Wins
</item 8>

<item 9>
African Footballer of the Year
DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- -- Chad -- -- Wins By Club ( edit ) Club Winners Runners - up Third places Olympique Marseille
Manchester City -- Chelsea 6 AS Monaco Barcelona Asante Kotoko -- Canon Yaoundé RCD Mallorca Inter Milan Liverpool Arsenal -- -- Paris SG -- Hafia Conacry TP
Mazembe -- Borussia Dortmund -- Sporting CP -- Al Ahly FC Porto Tonnerre Yaoundé -- RC Lens -- Milan -- Sporting CP -- GC Mascara -- Vitória de Setúbal --
Cercle Brugge -- -- CARA Brazzaville -- -- Leicester City -- -- AS Saint - Étienne -- -- Toulouse FC -- -- Parma -- -- FAR Rabat -- -- SCC Mohammédia -- -- JS
Saint - Pierroise -- -- Deportivo de La Coruña -- -- Espanyol -- -- Sevilla -- -- Espérance ST -- -- Bayern Munich -- -- MC Alger -- ASEC Mimosas -- Montpellier
-- Eintracht Frankfurt -- Shooting Stars -- Shanghai Shenhua -- -- Vita Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC
</item 9>

<item 10>
African Footballer of the Year
Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2013 - 07 - 02 . Jump
up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2016 - 09 - 29 .
</item 10>
=== END ITEM id=68d48186ed2c3326 ===

=== ITEM id=b58f0db8722baea5 ===
Gold sentences and facts:
Sentence 0: When Hoyt Axton performed the song to the group , two of the three main vocalists -- Danny Hutton and Cory Wells -- rejected the song , but Chuck Negron felt that the band needed a `` silly song '' to help bring the band back together as a working unit . Negron also felt that the song `` was n't even close to our best record , but it might have been one of our most honest . ''
  Fact 0.0: Danny Hutton and Cory Wells were two of the three main vocalists in the group.
  Fact 0.1: Danny Hutton rejected the song.
  Fact 0.2: Cory Wells rejected the song.
  Fact 0.3: Hoyt Axton performed the song to the group.
  Fact 0.4: Chuck Negron felt that the band needed a 'silly song' to help bring the band back together as a working unit.
  Fact 0.5: Chuck Negron felt the song was not even close to the band's best record.
  Fact 0.6: Chuck Negron felt the song might have been one of the band's most honest.
Sentence 1: The song , which has been described by members of Three Dog Night as a `` kid 's song '' and a `` silly song '' , topped the main singles charts in North America , was certified gold by the RIAA , and has since been covered by multiple artists .
  Fact 1.0: Members of Three Dog Night have described the song as a 'kid's song'.
  Fact 1.1: Members of Three Dog Night have described the song as a 'silly song'.
  Fact 1.2: The song topped the main singles charts in North America.
  Fact 1.3: The song was certified gold by the RIAA.
  Fact 1.4: The song has since been covered by multiple artists.

Retrieved items:
<item 1>
Graph query row:
song: Joy to the World
bandRelationship: RELATED_TO
band: Three Dog Night
writerRelationship: CREATED
songwriter: Hoyt Axton
source: Joy to the World (Three Dog Night song)
passage: Joy to the World (Three Dog Night song)
When the song hit number 1 on the US Billboard Hot 100 chart in 1971 , Axton and his mother , Mae Axton , became the first mother and son to each have written a
number one pop single in the rock era . Mae Axton co-wrote `` Heartbreak Hotel '' , which was the first number one hit for Elvis Presley . In a 1994 case ,
David P. Jackson filed suit claiming co-authorship of the song and alleging that Axton fraudulently claimed sole authorship . In the suit Jackson claimed that
Axton regularly credited him with co-authorship .
</item 1>

<item 2>
Joy to the World (Three Dog Night song)
The song , which has been described by members of Three Dog Night as a `` kid 's song '' and a `` silly song '' , topped the main singles charts in North
America , was certified gold by the RIAA , and has since been covered by multiple artists . The song is featured prominently in the film The Big Chill . It is
sung by a child character at the beginning and the Three Dog Night recording is played over the end credits . It is also played at the end of every Denver
Broncos home victory . Notable playings of this song after Broncos victories included then - Chicago Bears head coach Abe Gibron 's singing along with the song
in 1973 ; and at the end of Super Bowl XXXII , played at Qualcomm Stadium in San Diego . It was also played at the end of Super Bowl XXXIII at Pro Player ( now
Hard Rock ) Stadium in Miami Gardens , Florida and Super Bowl 50 in Santa Clara , California . Contents ( hide ) 1 Background and recording 2 Charts and awards
3 In popular culture 4 References 5 External links Background and Recording ( edit ) Some of the words are nonsensical .
</item 2>

<item 3>
Joy to the World (Three Dog Night song)
. Barrie & Jenkins . ISBN 0 - 214 - 20480 - 4 . Jump up ^ https://www.lambiek.net/artists/j/johnston_daniel.htm Jump up ^
https://www.lambiek.net/artists/j/johnston_daniel.htm Jump up ^ `` Big Lots TV Commercial , ' Joy : Recliners ' Song by Three Dog Night '' . ispot.tv .
Retrieved December 24 , 2017 .
</item 3>

<item 4>
Joy to the World (Three Dog Night song)
1965 -- 1975 ( 1993 ) 20th Century Masters -- The Millennium Collection : The Best of Three Dog Night ( 1999 ) Singles `` Nobody '' `` Try a Little Tenderness
'' `` One '' `` Easy to Be Hard '' `` Eli 's Coming '' `` Celebrate '' `` Mama Told Me ( Not to Come ) '' `` Out in the Country '' `` One Man Band '' `` Joy to
the World '' `` Liar '' `` An Old Fashioned Love Song '' `` Never Been to Spain '' `` The Family of Man '' `` Black & White '' `` Pieces of April '' `` Shambala
'' `` Let Me Serenade You '' `` The Show Must Go On '' `` Sure As I 'm Sittin ' Here '' `` Play Something Sweet ( Brickyard Blues ) '' `` Til the World Ends ''
`` Everybody Is a Masterpiece '' `` It 's a Jungle Out There '' Discography Book Billboard Year - End number one singles ( 1960 -- 1979 ) 1960 : `` Theme from A
Summer Place '' -- Percy Faith 1961 : `` Tossin ' and Turnin ' '' -- Bobby Lewis 1962 : ``
</item 4>

<item 5>
Joy to the World (Three Dog Night song)
ispot.tv . Retrieved December 24 , 2017 . External links ( edit ) Lyrics of this song at MetroLyrics Three Dog Night Danny Hutton Michael Allsup Paul Kingery
Pat Bautz Eddie Reasoner David Morgan Cory Wells Chuck Negron Jimmy Greenspoon Floyd Sneed Joe Schermie Skip Konte Al Ciner Gary Moon Richard Campbell Studio
albums Three Dog Night ( 1968 ) Suitable for Framing ( 1969 ) It Ai n't Easy ( 1970 ) Naturally ( 1970 ) Harmony ( 1971 ) Seven Separate Fools ( 1972 ) Cyan (
1973 ) Hard Labor ( 1974 ) Coming Down Your Way ( 1975 ) American Pastime ( 1976 ) It 's a Jungle ( 1983 ) Live albums Captured Live at the Forum ( 1969 )
Around the World with Three Dog Night ( 1973 ) Compilations Golden Bisquits ( 1971 ) Joy to the World : Their Greatest Hits ( 1975 ) The Best of 3 Dog Night (
1982 ) Celebrate : The Three Dog Night Story , 1965 -- 1975 (
</item 5>

<item 6>
Joy to the World (Three Dog Night song)
The single went on to sell 5 million copies worldwide , making it one of the best - selling singles of all time . In popular culture ( edit ) Singer -
songwriter Daniel Johnston likes this song , which inspired his own character Jeremiah the bullfrog , whom he often draws in his artwork and advertisements .
The frog is also featured on the cover of his album Hi , How Are You ( 1983 ) and has become an official music mascot for Johnston . It is sung by the son of
one of the main characters at the start of the film The Big Chill ( 1983 ) and is featured on the soundtrack . In The X-Files TV series ( Detour - S05E04 ) ,
Scully sings this song while she 's lost in a forest with Mulder to show him she is awake and he can rest . It is played during the end credits of the R - rated
animated film , Sausage Party ( 2016 ) , and later in the end credits of the PG - 13 - rated animated film A Very Spooky Night . It appears in 2017 and 2018 TV
commercials for Big Lots .
</item 6>

<item 7>
Joy to the World (Three Dog Night song)
Entertainment section . August 20 , 2008 . ISSN 1053 - 2889 . Retrieved April 5 , 2011 . Jump up ^ `` Jackson v. Axton '' . Law.justia.com . Retrieved 2016 - 10
- 04 . Jump up ^ `` Forum - 1970 ( ARIA Charts : Special Occasion Charts ) '' . Australian-charts.com . Archived from the original on 2016 - 10 - 20 . Retrieved
2017 - 05 - 07 . Jump up ^ `` 100 Singles '' ( PHP ) . RPM. 15 ( 15 ) . May 29 , 1971 . Retrieved April 5 , 2011 . Jump up ^ `` Musicline.de -- Three Dog Night
Single - Chartverfolgung '' ( in German ) . Media Control Charts . PhonoNet GmbH . Jump up ^ `` Dutchcharts.nl -- Three Dog Night -- Joy To The World '' ( in
Dutch ) . Single Top 100 .
</item 7>

<item 8>
Graph query row:
song: Joy to the World
bandRelationship: RELATED_TO
band: Three Dog Night
writerRelationship: CREATED
songwriter: Hoyt Axton
source: Joy to the World (Three Dog Night song)
passage: Joy to the World (Three Dog Night song)
It appears in 2017 and 2018 TV commercials for Big Lots . Preceded by `` Just My Imagination ( Running Away with Me ) '' by The Temptations Billboard Hot 100
number - one single April 17 , 1971 ( six weeks ) Succeeded by `` Brown Sugar '' by The Rolling Stones References ( edit ) ^ Jump up to : Celebrate : The Three
Dog Night Story , 1965 -- 1975 ( CD liner ) . Three Dog Night . United States : MCA Records . 1993 . pp. 27 , 30 , 31 . MCAD2 - 10956 . ^ Jump up to : Leaf ,
David ( 1993 ) . Celebrate : The Three Dog Night Story , 1965 -- 1975 ( CD liner ) . Three Dog Night . United States : MCA Records . p. 20 . MCAD2 - 10956 .
Jump up ^ `` Three Dog Night Headlines the Fair Tonight '' . Bainbridge Island Review . Entertainment section . August 20 , 2008 .
</item 8>

<item 9>
Graph query row:
song: Joy to the World
bandRelationship: RELATED_TO
band: Three Dog Night
writerRelationship: CREATED
songwriter: Hoyt Axton
source: Joy to the World (Three Dog Night song)
passage: Joy to the World (Three Dog Night song)
In the suit Jackson claimed that Axton regularly credited him with co-authorship . Charts and awards ( edit ) Weekly charts ( 1971 ) Peak position Australia KMR
8 Canadian Top Singles Germany ( Official German Charts ) 17 Netherlands ( Single Top 100 ) 25 New Zealand ( Listener ) 12 UK Singles ( Official Charts Company
) 24 US Billboard Hot 100 Year - end charts ( 1971 ) Rank Australia 50 Canadian Top Singles US Billboard Hot 100 The single had been out less than two months ,
when on April 9 , 1971 , `` Joy to the World '' was certified gold by the Recording Industry Association of America , for shipments of over 1,000,000 units
across the United States . The record was also given a Gold Leaf award by RPM magazine for sales of over a million units . The record won the award for the Best
Selling Hit Single Record by the National Association of Recording Merchandisers in March 1972 . It was also ranked by Billboard magazine as the year - end # 1
pop single of 1971 .
</item 9>

<item 10>
Joy to the World (Three Dog Night song)
Genre Rock Length 3 : 50 ( album ) 3 : 17 ( single ) Label Dunhill Songwriter ( s ) Hoyt Axton Producer ( s ) Richard Podolor Three Dog Night singles chronology
`` One Man Band '' ( 1970 ) `` Joy to the World '' ( 1971 ) `` Liar '' ( 1971 ) `` One Man Band '' ( 1970 ) `` Joy to the World '' ( 1971 ) `` Liar '' ( 1971 )
`` Joy to the World '' is a song written by Hoyt Axton , and made famous by the band Three Dog Night . The song is also popularly known by its opening lyric ,
`` Jeremiah was a bullfrog '' . Three Dog Night originally released the song on their fourth studio album , Naturally in November 1970 and subsequently released
an edited version of the song as a single in February 1971 .
</item 10>

<item 11>
Graph query row:
song: Joy to the World
bandRelationship: RELATED_TO
band: Three Dog Night
writerRelationship: CREATED
songwriter: Hoyt Axton
source: Joy to the World (Three Dog Night song)
passage: Joy to the World (Three Dog Night song)
Single Top 100 . Jump up ^ Flavour of New Zealand , 31 May 1971 Jump up ^ `` Official Singles Chart Top 100 '' . Official Charts Company . Jump up ^ `` Three
Dog Night Chart History ( Hot 100 ) '' . Billboard . Jump up ^ `` Forum - 1970 ( ARIA Charts : Special Occasion Charts ) '' . Australian-charts.com . Archived
from the original on 2016 - 10 - 20 . Retrieved 2017 - 05 - 07 . Jump up ^ `` 100 Top Singles of ' 71 '' ( PHP ) . RPM. 16 ( 20 ) : 5 . January 8 , 1972 .
Retrieved April 5 , 2011 . Jump up ^ `` Top 100 Hits of 1971 / Top 100 Songs of 1971 '' . Musicoutfitters.com . Retrieved 2016 - 10 - 04 . Jump up ^ `` American
single certifications -- Three Dog Night -- Joy to the World '' . Recording Industry Association of America .
</item 11>

<item 12>
Joy to the World (Three Dog Night song)
wikipedia.org/w/index.php?title=Joy_to_the_World_(Three_Dog_Night_song)&oldid=835489072 '' Categories : 1971 singles Billboard Hot 100 number - one singles
Three Dog Night songs Dunhill Records singles Songs written by Hoyt Axton Songs about animals Novelty songs RPM Top Singles number - one singles 1970 songs
Hidden categories : Articles needing additional references from June 2008 All articles needing additional references Articles with hAudio microformats
Singlechart usages for German Singlechart usages for Dutch100 Singlechart usages for UK Singlechart usages for Billboardhot100 Singlechart called without song
Talk Contents About Wikipedia 日本 語 Edit links This page was last edited on 9 April 2018 , at 00 : 34 . Text is available under the Creative Commons Attribution
- ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered
trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 12>

<item 13>
Joy to the World (Three Dog Night song)
Joy to the World ( Three Dog Night song ) - wikipedia Joy to the World ( Three Dog Night song ) Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
June 2008 ) ( Learn how and when to remove this template message ) `` Joy to the World '' 1971 French release Single by Three Dog Night from the album Naturally
B - side `` I Can Hear You Calling '' Released February 1971 ( 1971 - 02 ) Format 7 '' 45 RPM Recorded 1970 at American Recording Co .
</item 13>

<item 14>
Joy to the World (Three Dog Night song)
Axton wanted to persuade his record producers to record a new melody he had written and the producers asked him to sing any words to the tune . A member of
Three Dog Night said that the original lyrics to the song were `` Jeremiah was a prophet '' but `` no one liked that '' . When Hoyt Axton performed the song to
the group , two of the three main vocalists -- Danny Hutton and Cory Wells -- rejected the song , but Chuck Negron felt that the band needed a `` silly song ''
to help bring the band back together as a working unit . Negron also felt that the song `` was n't even close to our best record , but it might have been one of
our most honest . '' The song was recorded by Three Dog Night at American Recording Company , produced by Richard Podolor , and engineered by Bill Cooper .
Unlike most Three Dog Night songs recorded at that point , instead of having just the three main vocalists singing harmony , the song was recorded with all
seven members of the band singing . Drummer Floyd Sneed sings the deep lyric `` I wan na tell you '' towards the end of the song .
</item 14>
=== END ITEM id=b58f0db8722baea5 ===

=== ITEM id=2744802d730600ae ===
Gold sentences and facts:
Sentence 0: Late in the 20th century , lithium became an important component of battery electrolytes and electrodes , because of its high electrode potential . Because of its low atomic mass , it has a high charge - and power - to - weight ratio . A typical lithium - ion battery can generate approximately 3 volts per cell , compared with 2.1 volts for lead - acid or 1.5 volts for zinc - carbon cells . Lithium - ion batteries , which are rechargeable and have a high energy density , should not be confused with lithium batteries , which are disposable ( primary ) batteries with lithium or its compounds as the anode . Other rechargeable batteries that use lithium include the lithium - ion polymer battery , lithium iron phosphate battery , and the nanowire battery .
  Fact 0.0: Late in the 20th century, lithium became an important component of battery electrolytes and electrodes.
  Fact 0.1: Lithium became an important battery component because of its high electrode potential.
  Fact 0.2: Lithium has a low atomic mass.
  Fact 0.3: Lithium has a high charge-to-weight ratio because of its low atomic mass.
  Fact 0.4: Lithium has a high power-to-weight ratio because of its low atomic mass.
  Fact 0.5: A typical lithium-ion battery can generate approximately 3 volts per cell.
  Fact 0.6: Lead-acid batteries are compared at 2.1 volts.
  Fact 0.7: Zinc-carbon cells are compared at 1.5 volts.
  Fact 0.8: Lithium-ion batteries are rechargeable.
  Fact 0.9: Lithium-ion batteries have a high energy density.
  Fact 0.10: Lithium-ion batteries should not be confused with lithium batteries.
  Fact 0.11: Lithium batteries are disposable (primary) batteries.
  Fact 0.12: Lithium batteries have lithium or its compounds as the anode.
  Fact 0.13: The lithium-ion polymer battery is a rechargeable battery that uses lithium.
  Fact 0.14: The lithium iron phosphate battery is a rechargeable battery that uses lithium.
  Fact 0.15: The nanowire battery is a rechargeable battery that uses lithium.
Sentence 1: Lithium and its compounds have several industrial applications , including heat - resistant glass and ceramics , lithium grease lubricants , flux additives for iron , steel and aluminium production , lithium batteries , and lithium - ion batteries . These uses consume more than three quarters of lithium production .
  Fact 1.0: Lithium and its compounds have several industrial applications.
  Fact 1.1: Lithium and its compounds are used in heat-resistant glass and ceramics.
  Fact 1.2: Lithium grease lubricants are an industrial application of lithium.
  Fact 1.3: Lithium and its compounds are used as flux additives for iron, steel and aluminium production.
  Fact 1.4: Lithium batteries are an industrial application of lithium.
  Fact 1.5: Lithium-ion batteries are an industrial application of lithium.
  Fact 1.6: Uses of lithium and its compounds consume more than three quarters of lithium production.

Retrieved items:
<item 1>
Graph query row:
entity: lithium deuteride
relationship: RELATED_TO
fact: None
related: staged thermonuclear weapons
</item 1>

<item 2>
Lithium
We are not aware of any discoveries of lithium , ' he said . '' Lithia ( `` lithium brine '' ) is associated with tin mining areas in Cornwall , England and an
evaluation project from 400 - metre deep test boreholes is under consideration . If successful the hot brines will also provide geothermal energy to power the
lithium extraction and refining process . Biological ( edit ) Lithium is found in trace amount in numerous plants , plankton , and invertebrates , at
concentrations of 69 to 5,760 parts per billion ( ppb ) . In vertebrates the concentration is slightly lower , and nearly all vertebrate tissue and body fluids
contain lithium ranging from 21 to 763 ppb . Marine organisms tend to bioaccumulate lithium more than terrestrial organisms . Whether lithium has a
physiological role in any of these organisms is unknown . History ( edit ) Johan August Arfwedson is credited with the discovery of lithium in 1817 Petalite (
LiAlSi O ) was discovered in 1800 by the Brazilian chemist and statesman José Bonifácio de Andrada e Silva in a mine on the island of Utö , Sweden .
</item 2>

<item 3>
Lithium
000,000 Portugal 200 60,000 N / A Russia - - 1,000,000 Serbia - - 1,000,000 United States 38,000 6,900,000 Zimbabwe 900 23,000 100,000 + World total 32,500
14,000,000 46,900,000 See also : Lithium minerals Although lithium is widely distributed on Earth , it does not naturally occur in elemental form due to its
high reactivity . The total lithium content of seawater is very large and is estimated as 230 billion tonnes , where the element exists at a relatively constant
concentration of 0.14 to 0.25 parts per million ( ppm ) , or 25 micromolar ; higher concentrations approaching 7 ppm are found near hydrothermal vents .
Estimates for the Earth 's crustal content range from 20 to 70 ppm by weight . In keeping with its name , lithium forms a minor part of igneous rocks , with the
largest concentrations in granites . Granitic pegmatites also provide the greatest abundance of lithium - containing minerals , with spodumene and petalite
being the most commercially viable sources .
</item 3>

<item 4>
Lithium
Another significant mineral of lithium is lepidolite . A newer source for lithium is hectorite clay , the only active development of which is through the
Western Lithium Corporation in the United States . At 20 mg lithium per kg of Earth 's crust , lithium is the 25th most abundant element . According to the
Handbook of Lithium and Natural Calcium , `` Lithium is a comparatively rare element , although it is found in many rocks and some brines , but always in very
low concentrations . There are a fairly large number of both lithium mineral and brine deposits but only comparatively few of them are of actual or potential
commercial value . Many are very small , others are too low in grade . '' The US Geological Survey estimates that in 2010 , Chile had the largest reserves by
far ( 7.5 million tonnes ) and the highest annual production ( 8,800 tonnes ) . One of the largest reserve bases of lithium is in the Salar de Uyuni area of
Bolivia , which has 5.4 million tonnes . Other major suppliers include Australia , Argentina and China .
</item 4>

<item 5>
Graph query row:
entity: First isolation of lithium
relationship: RELATED_TO
fact: None
related: Lithium
</item 5>

<item 6>
Lithium
Lithium - wikipedia Lithium Jump to : navigation , search This article is about the chemical element . For the use of lithium as a medication , see Lithium (
medication ) . For other uses , see Lithium ( disambiguation ) . It has been suggested that Lithium as an investment be merged into this article . ( Discuss )
Proposed since June 2017 . Lithium , Li Lithium floating in oil General properties Pronunciation / ˈlɪθiəm / ​ ( LI - thee - əm ) Appearance silvery - white
Standard atomic weight ( A ) ( 7000693800000000000 ♠ 6.938 , 7000699700000000000 ♠ 6.997 ) conventional : 7000694000000000000 ♠ 6.94 Lithium in the periodic
table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium
Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium
</item 6>

<item 7>
Lithium
Lithium hydroxide is a strong base and , when heated with a fat , produces a soap made of lithium stearate . Lithium soap has the ability to thicken oils , and
it is used to manufacture all - purpose , high - temperature lubricating greases . Metallurgy ( edit ) Lithium ( e.g. as lithium carbonate ) is used as an
additive to continuous casting mould flux slags where it increases fluidity , a use which accounts for 5 % of global lithium use ( 2011 ) . Lithium compounds
are also used as additives ( fluxes ) to foundry sand for iron casting to reduce veining . Lithium ( as lithium fluoride ) is used as an additive to aluminium
smelters ( Hall -- Héroult process ) , reducing melting temperature and increasing electrical resistance , a use which accounts for 3 % of production ( 2011 ) .
When used as a flux for welding or soldering , metallic lithium promotes the fusing of metals during the process and eliminates the forming of oxides by
absorbing impurities .
</item 7>

<item 8>
Graph query row:
entity: lithium deuteride
relationship: RELATED_TO
fact: None
related: fusion fuel
</item 8>

<item 9>
Lithium
2 10 cm / mol ( 298 K ) Young 's modulus 4.9 GPa Shear modulus 4.2 GPa Bulk modulus 11 GPa Mohs hardness 0.6 Brinell hardness 5 MPa CAS Number 7439 - 93 - 2
History Discovery Johan August Arfwedson ( 1817 ) First isolation William Thomas Brande ( 1821 ) Main isotopes of lithium Isotope Abundance Half - life ( t )
Decay mode Product Li 5 % stable Li 95 % stable Li content may be as low as 3.75 % in natural samples . Li would therefore have a content of up to 96.25 % .
view talk references in Wikidata Lithium ( from Greek : λίθος , translit . lithos , lit . ' stone ' ) is a chemical element with symbol Li and atomic number 3 .
It is a soft , silvery - white alkali metal . Under standard conditions , it is the lightest metal and the lightest solid element . Like all alkali metals ,
lithium is highly reactive and flammable , and is stored in mineral oil .
</item 9>

<item 10>
Graph query row:
entity: Lithium as an investment
relationship: RELATED_TO
fact: None
related: Lithium
</item 10>

<item 11>
Lithium
The stockpiled lithium was depleted in lithium - 6 by 75 % , which was enough to affect the measured atomic weight of lithium in many standardized chemicals ,
and even the atomic weight of lithium in some `` natural sources '' of lithium ion which had been `` contaminated '' by lithium salts discharged from isotope
separation facilities , which had found its way into ground water . Lithium was used to decrease the melting temperature of glass and to improve the melting
behavior of aluminium oxide when using the Hall - Héroult process . These two uses dominated the market until the middle of the 1990s . After the end of the
nuclear arms race , the demand for lithium decreased and the sale of Department of Energy stockpiles on the open market further reduced prices . In the mid
1990s , several companies started to extract lithium from brine which proved to be a less expensive than underground or open - pit mining . Most of the mines
closed or shifted their focus to other materials because only the ore from zoned pegmatites could be mined for a competitive price . For example , the US mines
near Kings Mountain , North Carolina closed before the beginning of the 21st century .
</item 11>

<item 12>
Graph query row:
entity: transmutation of lithium atoms to helium
relationship: PARTICIPATED_IN
fact: None
related: Lithium
</item 12>

<item 13>
Lithium
When cut open , it exhibits a metallic luster , but moist air corrodes it quickly to a dull silvery gray , then black tarnish . It never occurs freely in nature
, but only in ( usually ionic ) compounds , such as pegmatitic minerals which were once the main source of lithium . Due to its solubility as an ion , it is
present in ocean water and is commonly obtained from brines . Lithium metal is isolated electrolytically from a mixture of lithium chloride and potassium
chloride . The nucleus of the lithium atom verges on instability , since the two stable lithium isotopes found in nature have among the lowest binding energies
per nucleon of all stable nuclides . Because of its relative nuclear instability , lithium is less common in the solar system than 25 of the first 32 chemical
elements even though its nuclei are very light : it is an exception to the trend that heavier nuclei are less common . For related reasons , lithium has
important uses in nuclear physics . The transmutation of lithium atoms to helium in 1932 was the first fully man - made nuclear reaction , and lithium deuteride
serves as a fusion fuel in staged thermonuclear weapons .
</item 13>

<item 14>
Graph query row:
entity: First isolation of lithium
relationship: PARTICIPATED_IN
fact: None
related: William Thomas Brande
</item 14>

<item 15>
Lithium
Extraction ( edit ) Lithium salts are extracted from water in mineral springs , brine pools , and brine deposits . Lithium is present in seawater , but
commercially viable methods of extraction have yet to be developed . Another potential source of lithium is the leachates of geothermal wells , which are
carried to the surface . Recovery of lithium has been demonstrated in the field ; the lithium is separated by simple filtration . The process and environmental
costs are primarily those of the already - operating well ; net environmental impacts may thus be positive . Uses ( edit ) Estimates of global lithium uses in
2015 Ceramics and glass ( 32 % ) Batteries ( 35 % ) Lubricating greases ( 9 % ) Continuous casting ( 5 % ) Air treatment ( 5 % ) Polymers ( 4 % ) Primary
aluminum production ( 1 % ) Pharmaceuticals ( < 1 % ) Other ( 9 % ) Ceramics and glass ( edit ) Lithium oxide is widely used as a flux for processing silica ,
reducing the melting point and viscosity of the material and leading to glazes with improved physical properties including low coefficients of thermal expansion
.
</item 15>

<item 16>
Lithium
Worldwide , this is one of the largest use for lithium compounds . Glazes containing lithium oxides are used for ovenware . Lithium carbonate ( Li CO ) is
generally used in this application because it converts to the oxide upon heating . Electrical and electronics ( edit ) Late in the 20th century , lithium became
an important component of battery electrolytes and electrodes , because of its high electrode potential . Because of its low atomic mass , it has a high charge
- and power - to - weight ratio . A typical lithium - ion battery can generate approximately 3 volts per cell , compared with 2.1 volts for lead - acid or 1.5
volts for zinc - carbon cells . Lithium - ion batteries , which are rechargeable and have a high energy density , should not be confused with lithium batteries
, which are disposable ( primary ) batteries with lithium or its compounds as the anode . Other rechargeable batteries that use lithium include the lithium -
ion polymer battery , lithium iron phosphate battery , and the nanowire battery . Lubricating greases ( edit ) Main article : Lithium grease The third most
common use of lithium is in greases .
</item 16>

<item 17>
Graph query row:
entity: Discovery of lithium
relationship: RELATED_TO
fact: None
related: Lithium
</item 17>

<item 18>
Graph query row:
entity: Discovery of lithium
relationship: PARTICIPATED_IN
fact: None
related: Johan August Arfwedson
</item 18>

<item 19>
Graph query row:
entity: Lithium (medication)
relationship: RELATED_TO
fact: None
related: Lithium
</item 19>

<item 20>
Graph query row:
entity: transmutation of lithium atoms to helium
relationship: RELATED_TO
fact: None
related: Helium
</item 20>
=== END ITEM id=2744802d730600ae ===
