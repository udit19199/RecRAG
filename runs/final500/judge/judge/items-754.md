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

This file has 6 items.

=== ITEM id=8bc1ed1a9178a91f ===
Gold sentences and facts:
Sentence 0: Within the oral mucosa ( a mucous membrane ) lining the mouth and also on the tongue and palates and mouth floor , are the minor salivary glands ; their secretions are mainly mucous and are innervated by the facial nerve ( the seventh cranial nerve ) . The glands also secrete amylase a first stage in the breakdown of food acting on the carbohydrate in the food to transform the starch content into maltose . There are other glands on the surface of the tongue that encircle taste buds on the back part of the tongue and these also produce lingual lipase . Lipase is a digestive enzyme that catalyses the hydrolysis of lipids ( fats ) . These glands are termed Von Ebner 's glands which have also been shown to have another function in the secretion of histatins which offer an early defense ( outside of the immune system ) against microbes in food , when it makes contact with these glands on the tongue tissue . Sensory information can stimulate the secretion of saliva providing the necessary fluid for the tongue to work with and also to ease swallowing of the food .
  Fact 0.0: The minor salivary glands are within the oral mucosa, which is a mucous membrane lining the mouth.
  Fact 0.1: The minor salivary glands are also located on the tongue, the palates, and the mouth floor.
  Fact 0.2: The secretions of the minor salivary glands are mainly mucous.
  Fact 0.3: The minor salivary glands are innervated by the facial nerve.
  Fact 0.4: The facial nerve is the seventh cranial nerve.
  Fact 0.5: The minor salivary glands also secrete amylase.
  Fact 0.6: Amylase is a first stage in the breakdown of food.
  Fact 0.7: Amylase acts on the carbohydrate in food to transform the starch content into maltose.
  Fact 0.8: Other glands on the surface of the tongue encircle taste buds on the back part of the tongue.
  Fact 0.9: These tongue glands also produce lingual lipase.
  Fact 0.10: Lipase is a digestive enzyme.
  Fact 0.11: Lipase catalyses the hydrolysis of lipids (fats).
  Fact 0.12: The tongue glands that produce lingual lipase are termed Von Ebner's glands.
  Fact 0.13: Von Ebner's glands have also been shown to have another function in the secretion of histatins.
  Fact 0.14: Histatins offer an early defense against microbes in food, outside of the immune system.
  Fact 0.15: Histatins act against microbes in food when the food makes contact with the glands on the tongue tissue.
  Fact 0.16: Sensory information can stimulate the secretion of saliva.
  Fact 0.17: Saliva provides the necessary fluid for the tongue to work with.
  Fact 0.18: Saliva also eases swallowing of the food.

Retrieved items:
<item 1>
Human digestive system
This produces a bolus which can be swallowed down the esophagus and into the stomach . Here it is mixed with gastric juice until it passes into the duodenum
where it is mixed with a number of enzymes produced by the pancreas . Saliva also contains a catalytic enzyme called amylase which starts to act on food in the
mouth . Another digestive enzyme called lingual lipase is secreted by some of the lingual papillae on the tongue and also from serous glands in the main
salivary glands . Digestion is helped by the mastication of food by the teeth and also by the muscular actions of peristalsis and segmentation contractions .
Gastric juice in the stomach is essential for the continuation of digestion as is the production of mucus in the stomach . Peristalsis is the rhythmic
contraction of muscles that begins in the esophagus and continues along the wall of the stomach and the rest of the gastrointestinal tract . This initially
results in the production of chyme which when fully broken down in the small intestine is absorbed as chyle into the lymphatic system . Most of the digestion of
food takes place in the small intestine .
</item 1>

<item 2>
Small intestine
The three major classes of nutrients that undergo digestion are proteins , lipids ( fats ) and carbohydrates : Proteins are degraded into small peptides and
amino acids before absorption . Chemical breakdown begins in the stomach and continues in the small intestine . Proteolytic enzymes , including trypsin and
chymotrypsin , are secreted by the pancreas and cleave proteins into smaller peptides . Carboxypeptidase , which is a pancreatic brush border enzyme , splits
one amino acid at a time . Aminopeptidase and dipeptidase free the end amino acid products . Lipids ( fats ) are degraded into fatty acids and glycerol .
Pancreatic lipase breaks down triglycerides into free fatty acids and monoglycerides . Pancreatic lipase works with the help of the salts from the bile secreted
by the liver and stored in the gall bladder . Bile salts attach to triglycerides to help emulsify them , which aids access by pancreatic lipase .
</item 2>

<item 3>
Human digestive system
The large intestine includes the rectum and anal canal . Small intestine ( edit ) Lower GI tract - 3 ) Small intestine ; 5 ) Cecum ; 6 ) Large intestine
Duodenum Food starts to arrive in the small intestine one hour after it is eaten , and after two hours the stomach has emptied . Until this time the food is
termed a bolus . It then becomes the partially digested semi-liquid termed chyme . In the small intestine , the pH becomes crucial ; it needs to be finely
balanced in order to activate digestive enzymes . The chyme is very acidic , with a low pH , having been released from the stomach and needs to be made much
more alkaline . This is achieved in the duodenum by the addition of bile from the gall bladder combined with the bicarbonate secretions from the pancreatic duct
and also from secretions of bicarbonate - rich mucus from duodenal glands known as Brunner 's glands . The chyme arrives in the intestines having been released
from the stomach through the opening of the pyloric sphincter . The resulting alkaline fluid mix neutralises the gastric acid which would damage the lining of
the intestine .
</item 3>

<item 4>
Small intestine
This is because the cellulose is made out of beta - glucose , making the inter-monosaccharidal bindings different from the ones present in starch , which
consists of alpha - glucose . Humans lack the enzyme for splitting the beta - glucose - bonds , something reserved for herbivores and bacteria from the large
intestine . Absorption ( edit ) Digested food is now able to pass into the blood vessels in the wall of the intestine through either diffusion or active
transport . The small intestine is the site where most of the nutrients from ingested food are absorbed . The inner wall , or mucosa , of the small intestine is
lined with simple columnar epithelial tissue . Structurally , the mucosa is covered in wrinkles or folds called plicae circulares , which are considered
permanent features in the wall of the organ . They are distinct from rugae which are considered non-permanent or temporary allowing for distention and
contraction . From the plicae circulares project microscopic finger - like pieces of tissue called villi ( Latin for `` shaggy hair '' ) . The individual
epithelial cells also have finger - like projections known as microvilli .
</item 4>

<item 5>
Human digestive system
Chyle is then transported through the lymphatic system to the rest of the body . The suspensory muscle marks the end of the duodenum and the division between
the upper gastrointestinal tract and the lower GI tract . The digestive tract continues as the jejunum which continues as the ileum . The jejunum , the
midsection of the small intestine contains circular folds , flaps of doubled mucosal membrane which partially encircle and sometimes completely encircle the
lumen of the intestine . These folds together with villi serve to increase the surface area of the jejunum enabling an increased absorption of digested sugars ,
amino acids and fatty acids into the bloodstream . The circular folds also slow the passage of food giving more time for nutrients to be absorbed . The last
part of the small intestine is the ileum . This also contains villi and vitamin B12 ; bile acids and any residue nutrients are absorbed here . When the chyme is
exhausted of its nutrients the remaining waste material changes into the semi-solids called feces , which pass to the large intestine , where bacteria in the
gut flora further break down residual proteins and starches .
</item 5>

<item 6>
Small intestine
This occurs because the lipase is water - soluble but the fatty triglycerides are hydrophobic and tend to orient towards each other and away from the watery
intestinal surroundings . The bile salts emulsify the triglycerides in the watery surroundings until the lipase can break them into the smaller components that
are able to enter the villi for absorption . Some carbohydrates are degraded into simple sugars , or monosaccharides ( e.g. , glucose ) . Pancreatic amylase
breaks down some carbohydrates ( notably starch ) into oligosaccharides . Other carbohydrates pass undigested into the large intestine and further handling by
intestinal bacteria . Brush border enzymes take over from there . The most important brush border enzymes are dextrinase and glucoamylase , which further break
down oligosaccharides . Other brush border enzymes are maltase , sucrase and lactase . Lactase is absent in some adult humans and , for them , lactose , like
most poly - saccharides , is not digested in the small intestine . Some carbohydrates , such as cellulose , are not digested at all , despite being made of
multiple glucose units .
</item 6>

<item 7>
Human digestive system
The mucus component lubricates the walls of the intestine . When the digested food particles are reduced enough in size and composition , they can be absorbed
by the intestinal wall and carried to the bloodstream . The first receptacle for this chyme is the duodenal bulb . From here it passes into the first of the
three sections of the small intestine , the duodenum . ( The next section is the jejunum and the third is the ileum ) . The duodenum is the first and shortest
section of the small intestine . It is a hollow , jointed C - shaped tube connecting the stomach to the jejunum . It starts at the duodenal bulb and ends at the
suspensory muscle of duodenum . The attachment of the suspensory muscle to the diaphragm is thought to help the passage of food by making a wider angle at its
attachment . Most food digestion takes place in the small intestine . Segmentation contractions act to mix and move the chyme more slowly in the small intestine
allowing more time for absorption ( and these continue in the large intestine ) .
</item 7>

<item 8>
Small intestine
The parts of the intestine are as follows : Layer Duodenum Jejunum Ileum Serosa 1st part serosa , 2nd -- 4th adventitia Normal Normal Muscularis externa
Longitudinal and circular layers , with Auerbach 's ( myenteric ) plexus in between Same as duodenum Same as duodenum Submucosa Brunner 's glands and Meissner
's ( submucosal ) plexus No BG No BG Mucosa : muscularis mucosae Normal Normal Normal Mucosa : lamina propria No PP No PP Peyer 's patches Mucosa : intestinal
epithelium Simple columnar . Contains goblet cells , Paneth cells Similar to duodenum ? Development ( edit ) See also : Development of the digestive system The
small intestine develops from the midgut of the primitive gut tube . By the fifth week of embryological life , the ileum begins to grow longer at a very fast
rate , forming a U-shaped fold called the primary intestinal loop . The loop grows so fast in length that it outgrows the abdomen and protrudes through the
umbilicus .
</item 8>

<item 9>
Human digestive system
This breaks down fats to some degree though is not as efficient as the pancreatic lipase . The pylorus , the lowest section of the stomach which attaches to the
duodenum via the pyloric canal , contains countless glands which secrete digestive enzymes including gastrin . After an hour or two , a thick semi-liquid called
chyme is produced . When the pyloric sphincter , or valve opens , chyme enters the duodenum where it mixes further with digestive enzymes from the pancreas ,
and then passes through the small intestine , where digestion continues . When the chyme is fully digested , it is absorbed into the blood . 95 % of absorption
of nutrients occurs in the small intestine . Water and minerals are reabsorbed back into the blood in the colon of the large intestine , where the environment
is slightly acidic . Some vitamins , such as biotin and vitamin K produced by bacteria in the gut flora of the colon are also absorbed . The parietal cells in
the fundus of the stomach , produce a glycoprotein called intrinsic factor which is essential for the absorption of vitamin B12 .
</item 9>

<item 10>
Small intestine
By week 10 , the loop retracts back into the abdomen . Between weeks six and ten the small intestine rotates anticlockwise , as viewed from the front of the
embryo . It rotates a further 180 degrees after it has moved back into the abdomen . This process creates the twisted shape of the large intestine . Function (
edit ) Food from the stomach is allowed into the duodenum through the pylorus by a muscle called the pyloric sphincter . Digestion ( edit ) The small intestine
is where most chemical digestion takes place . Many of the digestive enzymes that act in the small intestine are secreted by the pancreas and liver and enter
the small intestine via the pancreatic duct . Pancreatic enzymes and bile from the gallbladder enter the small intestine in response to the hormone
cholecystokinin , which is produced in the small intestine in response to the presence of nutrients . Secretin , another hormone produced in the small intestine
, causes additional effects on the pancreas , where it promotes the release of bicarbonate into the duodenum in order to neutralize the potentially harmful acid
coming from the stomach .
</item 10>
=== END ITEM id=8bc1ed1a9178a91f ===

=== ITEM id=c15a7b3298898067 ===
Gold sentences and facts:
Sentence 0: In vertebrates , the gallbladder is a small hollow organ where bile is stored and concentrated before it is released into the small intestine . In humans , the pear - shaped gallbladder lies beneath the liver , although the structure and position of the gallbladder can vary significantly among animal species . It receives and stores bile , produced by the liver , via the common hepatic duct , and releases it via the common bile duct into the duodenum , where the bile helps in the digestion of fats .
  Fact 0.0: In vertebrates, the gallbladder is a small hollow organ.
  Fact 0.1: In vertebrates, bile is stored in the gallbladder before it is released into the small intestine.
  Fact 0.2: In vertebrates, bile is concentrated in the gallbladder before it is released into the small intestine.
  Fact 0.3: In humans, the gallbladder is pear-shaped.
  Fact 0.4: In humans, the gallbladder lies beneath the liver.
  Fact 0.5: The structure and position of the gallbladder can vary significantly among animal species.
  Fact 0.6: The gallbladder receives bile produced by the liver via the common hepatic duct.
  Fact 0.7: The gallbladder stores bile produced by the liver via the common hepatic duct.
  Fact 0.8: The gallbladder releases bile via the common bile duct into the duodenum.
  Fact 0.9: Bile helps in the digestion of fats in the duodenum.

Retrieved items:
<item 1>
Gallbladder
This sits on a thin layer of connective tissue , the lamina propria . The mucosa is curved and collected into tiny outpouchings called rugae . A muscular layer
sits beneath the mucosa . This is formed by smooth muscle , with fibres that lie in longitudinal , oblique and transverse directions , and are not arranged in
separate layers . The muscle fibres here contract to expel bile from the gallbladder . A distinctive feature of the gallbladder is the presence of Rokitansky --
Aschoff sinuses , deep outpouchings of the mucosa that can extend through the muscular layer . The muscular layer is surrounded by a layer of connective and fat
tissue . The outer layer of the fundus of gallbladder , and the surfaces not in contact with the liver , are covered by a thick serosa , which is exposed to the
peritoneum . The serosa contains blood vessels and lymphatics . The surfaces in contact with the liver are covered in connective tissue . Variation ( edit ) The
gallbladder varies in size , shape , and position between different people .
</item 1>

<item 2>
Gallbladder
5 ( 1 ) : 89 -- 94 . Books editor - in - chief , Susan Standring ; section editors , Neil R. Borley ; et al. ( 2008 ) . Gray 's anatomy : the anatomical basis
of clinical practice ( 40th ed . ) . London : Churchill Livingstone . ISBN 978 - 0 - 8089 - 2371 - 8 . CS1 maint : Explicit use of et al. ( link ) CS1 maint :
Multiple names : authors list ( link ) CS1 maint : Extra text : authors list ( link ) External links ( edit ) Wikimedia Commons has media related to Gallbladder
. Diagram of Human Stomach and Gallbladder -- Human Anatomy Online dd , MyHealthScore.com . ( hide ) Anatomy of the liver ,
</item 2>

<item 3>
Gallbladder
doi : 10.1080 / 136518202753598726 . Jump up ^ Naganuma , S. ; Ishida , H. ; Konno , K. ; Hamashima , Y. ; Hoshino , T. ; Naganuma , H. ; Komatsuda , T. ;
Ohyama , Y. ; Yamada , N. ; Ishida , J. ; Masamune , O. ( March 6 , 2014 ) . `` Sonographic findings of anomalous position of the gallbladder '' . Abdominal
Imaging. 23 ( 1 ) : 67 -- 72 . doi : 10.1007 / s002619900287 . PMID 9437066 . Jump up ^ Meilstrup JW ; Hopper KD ; Thieme GA ( December 1991 ) . `` Imaging of
gallbladder variants '' . AJR Am J Roentgenol. 157 ( 6 ) : 1205 -- 8 . doi : 10.2214 / ajr. 157.6. 1950867 . PMID 1950867 .
</item 3>

<item 4>
Gallbladder
`` Anatomy of the Murine Hepatobiliary System : A Whole - Organ - Level Analysis Using a Transparency Method '' . The Anatomical Record . 299 ( 2 ) : 161 -- 172
. doi : 10.1002 / ar. 23287 . PMID 26559382 . Jump up ^ Romer , Alfred Sherwood ; Parsons , Thomas S. ( 1977 ) . The Vertebrate Body . Philadelphia , PA : Holt
- Saunders International . p. 355 . ISBN 0 - 03 - 910284 - X . Jump up ^ Actman , Jani ( 5 May 2016 ) . `` Inside the Disturbing World of Bear - Bile Farming ''
. National Geographic . Retrieved 23 October 2017 . Jump up ^ Hance , Jeremy ( 9 April 2015 ) . `` Is the end of ' house of horror ' bear bile factories in
sight ? '' . The Guardian . Retrieved 23 October 2017 .
</item 4>

<item 5>
Gallbladder
In adults , the gallbladder measures approximately 7 to 10 centimetres ( 2.8 to 3.9 inches ) in length and 4 centimetres ( 1.6 in ) in diameter when fully
distended . The gallbladder has a capacity of about 50 millilitres ( 1.8 imperial fluid ounces ) . The gallbladder is shaped like a pear , with its tip opening
into the cystic duct . The gallbladder is divided into three sections : the fundus , body , and neck . The fundus is the rounded base , angled so that it faces
the abdominal wall . The body lies in a depression in the surface of the lower liver . The neck tapers and is continuous with the cystic duct , part of the
biliary tree . The gallbladder fossa , against which the fundus and body of the gallbladder lie , is found beneath the junction of hepatic segments IVB and V .
The cystic duct unites with the common hepatic duct to become the common bile duct .
</item 5>

<item 6>
Gallbladder
Function ( edit ) 1 . Bile ducts : 2 . Intrahepatic bile ducts , 3 . Left and right hepatic ducts , 4 . Common hepatic duct , 5 . Cystic duct , 6 . Common bile
duct , 7 . Ampulla of Vater , 8 . Major duodenal papilla 9 . Gallbladder , 10 -- 11 . Right and left lobes of liver . 12 . Spleen . 13 . Esophagus. 14 . Stomach
. 15 . Pancreas : 16 . Accessory pancreatic duct , 17 . Pancreatic duct . 18 . Small intestine : 19 . Duodenum , 20 . Jejunum 21 -- 22 . Right and left kidneys
. The front border of the liver has been lifted up ( brown arrow ) . The main purpose of the gallbladder is to store bile , also called gall , needed for the
digestion of fats in food .
</item 6>

<item 7>
Gallbladder
Gallbladder - Wikipedia Gallbladder This is the latest accepted revision , reviewed on 18 January 2018 . Jump to : navigation , search Gallbladder The
gallbladder sits beneath the liver . Details Precursor Foregut System Digestive system Artery Cystic artery Vein Cystic vein Nerve Celiac ganglia , Vagus nerve
Identifiers Latin Vesica biliaris , vesica fellea MeSH A03. 159.439 Dorlands / Elsevier 12383343 TA A05. 8.02. 001 FMA 7202 Anatomical terminology ( edit on
Wikidata ) In vertebrates , the gallbladder is a small hollow organ where bile is stored and concentrated before it is released into the small intestine . In
humans , the pear - shaped gallbladder lies beneath the liver , although the structure and position of the gallbladder can vary significantly among animal
species . It receives and stores bile , produced by the liver , via the common hepatic duct , and releases it via the common bile duct into the duodenum , where
the bile helps in the digestion of fats .
</item 7>

<item 8>
Gallbladder
Early in development , the human embryo has three germ layers and abuts an embryonic yolk sac . During the second week of embryogenesis , as the embryo grows ,
it begins to surround and envelop portions of this sac . The enveloped portions form the basis for the adult gastrointestinal tract . Sections of this foregut
begin to differentiate into the organs of the gastrointestinal tract , such as the esophagus , stomach , and intestines . During the fourth week of
embryological development , the stomach rotates . The stomach , originally lying in the midline of the embryo , rotates so that its body is on the left . This
rotation also affects the part of the gastrointestinal tube immediately below the stomach , which will go on to become the duodenum . By the end of the fourth
week , the developing duodenum begins to spout a small outpouching on its right side , the hepatic diverticulum , which will go on to become the biliary tree .
Just below this is a second outpouching , known as the cystic diverticulum , that will eventually develop into the gallbladder . Function ( edit ) 1 .
</item 8>

<item 9>
Gallbladder
30 . Archived from the original on April 1 , 2008 . Retrieved June 29 , 2007 . ^ Jump up to : Gray 's Anatomy 2008 , p. 1187 - 81 . Jump up ^ Jon W. Meilstrup (
1994 ) . Imaging Atlas of the Normal Gallbladder and Its Variants . Boca Raton : CRC Press . p. 4 . ISBN 0 - 8493 - 4788 - 2 . Jump up ^ Nagral , Sanjay ( 2005
) . `` Anatomy relevant to cholecystectomy '' . Journal of Minimal Access Surgery . 1 ( 2 ) : 53 . doi : 10.4103 / 0972 - 9941.16527 . Jump up ^ Shakelford 's
Surgery of Alimentary Tract , ed. 7 . 2013 ^ Jump up to : Young , Barbara ; et al. ( 2006 ) . Wheater 's functional histology : a text and colour atlas ( 5th ed
. ) . ( Edinburgh ? )
</item 9>

<item 10>
Gallbladder
( hide ) Anatomy of the liver , pancreas and biliary tree Liver Bare area Cantlie line Ligamentum venosum Porta hepatis Round ligament Lobes of liver Caudate
Quadrate Fibrous capsule of Glisson Perisinusoidal space Liver sinusoid Periportal space Portal triad Lobules of liver Liver segment Microanatomy Hepatocyte
Hepatic stellate cell Kupffer cell Biliary tract Bile ducts Intrahepatic bile ducts Bile canaliculus Canals of Hering Interlobular Left hepatic duct Right
hepatic duct Common hepatic duct Gallbladder Cystic duct Common bile duct Ampulla of Vater Sphincter of Oddi Cells Cholecystocyte Pancreas Gross Tail Body Neck
Head ( Uncinate process ) Ducts main accessory Microanatomy Pancreatic islets Centroacinar cell Stellate cell GND : 4071544 - 9 NDL : 00572670 Retrieved from ``
https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=c15a7b3298898067 ===

=== ITEM id=991a72b506ea6782 ===
Gold sentences and facts:
Sentence 0: Third Generation ( GK Series ) Overview Production 2013 -- present Model years 2014 -- present Assembly Yorii , Japan Sumaré , Brazil Ayutthaya , Thailand Pingtung , Taiwan Karawang , Indonesia Malacca , Malaysia Guangzhou , China ( Guangqi Honda ) Celaya , Guanajuato , Mexico ( from Feb 2014 ) Tapukara , India ( from Jan 2015 ) Designer Riku Wada ( 2010 ) Body and chassis Related Honda City Honda HR - V Honda Freed Honda Shuttle Acura CDX Powertrain Engine 1.2 L L12B i - VTEC I4 petrol 1.3 L L13B Earth Dreams i - VTEC I4 1.5 L L15A I4 ( petrol ) 1.5 L L15B1 I4 ( petrol ) 1.5 L L15B Earth Dreams i - VTEC I4 1.5 L i - DTEC diesel Transmission 5 - speed manual ( China Only ) 6 - speed manual 7 - speed semi-automatic ( DCT ) CVT automatic Dimensions Wheelbase 2,530 mm ( 99.6 in ) Length 3,955 mm ( 155.7 in ) Width 1,695 mm ( 66.7 in ) Height 1,525 -- 1,550 mm ( 60.0 -- 61.0 in ) Curb weight 1,020 -- 1,080 kg ( 2,249 -- 2,381 lb ) 1,080 -- 1,170 kg ( 2,381 -- 2,579 lb ) ( Hybrid )
  Fact 0.0: The Third Generation (GK Series) has production from 2013 to the present.
  Fact 0.1: The Third Generation (GK Series) has model years from 2014 to the present.
  Fact 0.2: The Third Generation (GK Series) is assembled in Yorii, Japan.
  Fact 0.3: The Third Generation (GK Series) is assembled in Sumaré, Brazil.
  Fact 0.4: The Third Generation (GK Series) is assembled in Ayutthaya, Thailand.
  Fact 0.5: The Third Generation (GK Series) is assembled in Pingtung, Taiwan.
  Fact 0.6: The Third Generation (GK Series) is assembled in Karawang, Indonesia.
  Fact 0.7: The Third Generation (GK Series) is assembled in Malacca, Malaysia.
  Fact 0.8: The Third Generation (GK Series) is assembled in Guangzhou, China (Guangqi Honda).
  Fact 0.9: The Third Generation (GK Series) is assembled in Celaya, Guanajuato, Mexico, from February 2014.
  Fact 0.10: The Third Generation (GK Series) is assembled in Tapukara, India, from January 2015.
  Fact 0.11: The Third Generation (GK Series) was designed by Riku Wada in 2010.
  Fact 0.12: The Third Generation (GK Series) is related to the Honda City.
  Fact 0.13: The Third Generation (GK Series) is related to the Honda HR-V.
  Fact 0.14: The Third Generation (GK Series) is related to the Honda Freed.
  Fact 0.15: The Third Generation (GK Series) is related to the Honda Shuttle.
  Fact 0.16: The Third Generation (GK Series) is related to the Acura CDX.
  Fact 0.17: The Third Generation (GK Series) has a 1.2 L L12B i-VTEC I4 petrol engine.
  Fact 0.18: The Third Generation (GK Series) has a 1.3 L L13B Earth Dreams i-VTEC I4 engine.
  Fact 0.19: The Third Generation (GK Series) has a 1.5 L L15A I4 petrol engine.
  Fact 0.20: The Third Generation (GK Series) has a 1.5 L L15B1 I4 petrol engine.
  Fact 0.21: The Third Generation (GK Series) has a 1.5 L L15B Earth Dreams i-VTEC I4 engine.
  Fact 0.22: The Third Generation (GK Series) has a 1.5 L i-DTEC diesel engine.
  Fact 0.23: The 5-speed manual transmission is available in China only.
  Fact 0.24: The Third Generation (GK Series) has a 6-speed manual transmission.
  Fact 0.25: The Third Generation (GK Series) has a 7-speed semi-automatic (DCT) transmission.
  Fact 0.26: The Third Generation (GK Series) has a CVT automatic transmission.
  Fact 0.27: The Third Generation (GK Series) has a wheelbase of 2,530 mm (99.6 in).
  Fact 0.28: The Third Generation (GK Series) has a length of 3,955 mm (155.7 in).
  Fact 0.29: The Third Generation (GK Series) has a width of 1,695 mm (66.7 in).
  Fact 0.30: The Third Generation (GK Series) has a height of 1,525 to 1,550 mm (60.0 to 61.0 in).
  Fact 0.31: The Third Generation (GK Series) has a curb weight of 1,020 to 1,080 kg (2,249 to 2,381 lb).
  Fact 0.32: The hybrid version of the Third Generation (GK Series) has a curb weight of 1,080 to 1,170 kg (2,381 to 2,579 lb).

Retrieved items:
<item 1>
Honda Fit
United States and Canada 2007 Honda Fit base and Sport ( U.S. ) The Fit went on sale in the United States and Canada in April 2006 for model year 2007 ,
replacing the Civic hatchback for North America and becoming that market 's chief entry - level subcompact . The Civic continues to grow in size and price and
was reclassified as a compact ( larger than a subcompact ) in 2001 . In keeping with Honda 's safety initiative , the Fit was offered in North America with
power windows , standard side airbags , side - curtain airbags , and ABS , in addition to the mandatory front airbags . Due to differing safety regulations ,
North American Fits have larger bumpers , and a longer front clip , than the rest of the world , resulting in a slight increase in overall length . For the 2008
model , US models feature the Tire Pressure Monitoring System ( TPMS ) to alert the driver of low pressure in any of the four stock wheels . Unlike some systems
, the system in the 2008 Fit notifies the driver that a tyre is low , but not which tyre .
</item 1>

<item 2>
Honda Fit
A total of 56,432 Fits were sold in the United States in 2007 . The 2008 Honda Fit was largely unchanged from the 2007 with the exception of the TPMS and new
windshield wipers that abandon the large rubber blade design to a more traditional rigid plastic with thin rubber blade design . MSRP was also increased by
roughly US $100 . Total sales in the United States for the year were 79,794 , an increase of almost 41 % over previous year . Reception Honda marketed the Fit
as `` Small is the New Big '' . Car and Driver magazine described that `` ( n ) ot only is it endowed with an impressive array of standard features and an
upscale interior , but its dynamics rival the responses of cars with much fancier pedigrees and price tags . '' The magazine even cited its lane - change number
of 71.4 mph edged the best run by a Ferrari F430 Spider F1 tested . And the car was included in Car and Driver 's 10 Best Lists for three times . In May 2006 ,
Fit was placed first in a Car and Driver comparison of seven economy cars that included Nissan Versa and Toyota Yaris .
</item 2>

<item 3>
Honda Fit
Retrieved 2010 - 01 - 20 . Jump up ^ `` Honda Adds a 1.5 L Model to the Popular `` Fit '' Small Car Line - up `` . Honda . 2002 - 09 - 12 . Archived from the
original on 2016 - 03 - 02 . Retrieved 2010 - 01 - 20 . Jump up ^ `` Top Gear Survey 2006 - Honda Jazz '' . Archived from the original on 2007 - 10 - 11 .
Retrieved 2008 - 01 - 14 . Jump up ^ `` Honda Begins Production of `` Fit '' Hatchback in China `` . Honda . 2004 - 09 - 13 . Archived from the original on 2016
- 03 - 02 . Retrieved 2010 - 01 - 20 . Jump up ^ `` 2008 Honda Fit Review '' . JB car pages . Retrieved 2009 - 03 - 09 . Jump up ^ Chris Vander Doelen , Windsor
Star ( April 5 , 2007 ) .
</item 3>

<item 4>
Honda Fit
525 -- 1,550 mm ( 60.0 -- 61.0 in ) Curb weight 1,020 -- 1,080 kg ( 2,249 -- 2,381 lb ) 1,080 -- 1,170 kg ( 2,381 -- 2,579 lb ) ( Hybrid ) The third generation
Fit retains the overall design concept of previous generations , notably the center - located fuel tank and multi-configuration interior concept , marketed as
Magic Seating . Honda 's all - new `` Global Small Car Platform '' employing ultra-high strength 780 MPa yield steel for 27 % of its bodywork -- and a shorter
overall length ( - 1.6 inches ) , longer wheelbase ( + 1.2 inch ) increased rear legroom ( + 4.8 inches ) and increased passenger volume ( + 4.9 cu ft )
compared to its previous generation . Body panels are both welded and bolted to the frame in a hybrid monocoque and spaceframe fusion -- and rear torsion beam
suspension is more compact , no longer using an anti-sway bar to maximize interior and cargo space . The HR - V shares its platform with the Fit .
</item 4>

<item 5>
Honda Fit
Gawker Media . Retrieved 2008 - 03 - 19 . Jump up ^ Bell , Damon . `` The Consumer Guide to 2009 Subcompact Cars '' . Consumer Guide Automotive . HowStuffWorks
. Archived from the original on 2009 - 08 - 30 . Retrieved 2009 - 08 - 25 . The most expensive subcompact , the top - trim - level Honda Fit with an optional
navigation system , has a still - modest base price of $18,760 . Jump up ^ `` Honda Fit production could shift to Ohio '' . The Columbus Dispatch . Archived
from the original on 2013 - 01 - 21 . Retrieved 2009 - 10 - 22 . Jump up ^ `` 2011 Honda Fit Receives More Standard Features ; VSA Added to All Models '' .
theautochannel.com . Jump up ^ `` 2009 Honda Fit Review '' . JB car pages . Retrieved 2009 - 03 - 09 . Jump up ^ `` 2012 Honda Fit Receives Updates and More
Features '' ( Press release ) .
</item 5>

<item 6>
Honda Fit
3 L L13A I4 ( petrol ) 1.5 L L15A I4 ( petrol ) Transmission 5 - speed manual 6 - speed manual 5 - speed automatic CVT automatic ( Japan ) Dimensions Wheelbase
2,500 mm ( 98.4 in ) Length 3,985 mm ( 156.9 in ) ( 2009 -- 11 ) 4,100 mm ( 161.4 in ) ( 2012 -- 13 ) 4,410 mm ( 173.6 in ) ( Fit Shuttle ) Width 1,695 mm (
66.7 in ) Height 1,525 mm ( 60.0 in ) 1,540 mm ( 60.6 in ) ( Fit Shuttle ) Curb weight 1,070 kg ( 2,359 lb ) The second generation Fit debuted on 17 October
2007 at the Tokyo Motor Show . At its introduction in 2007 , it won the Car of the Year Japan Award for the second time . The vehicle offered a longer wheelbase
than its predecessor and is wider and longer overall . Overall height is unchanged , while interior height increased by . 3 in .
</item 6>

<item 7>
Honda Fit
National Highway Traffic Safety Administration ( NHTSA ) crash test ratings . The engine is a 1.5 L ( 92 cu in ) i - VTEC 4 - cylinder with 117 horsepower at
6600 rpm mated to a five - speed manual transmission or a five - speed automatic transmission . The second generation Honda Fit went on sale in North America
around August 26 , 2008 , for the 2009 model year . It was considered the most expensive within the subcompact class when equipped as such , but was also
described as modestly priced . It was reported in September 2009 that Honda is considering producing the Fit in North America in response to the increasing
strength of the Japanese yen against the U.S. dollar . For the 2011 models , Vehicle Stability Assist , cruise control , remote entry and USB audio interface
became standard across all trim levels . In addition , the Fit Sport now comes with carpeted floor mats .
</item 7>

<item 8>
Honda Fit
Retrieved 2009 - 12 - 11 . Jump up ^ James Healey ( Feb 28 , 2011 ) . `` Consumer Reports lauds Honda Fit for value , rips Jeep Wrangler '' . USA Today . Jump
up ^ `` Honda Fit tops list of best value cars '' . The Boston Globe . Mar 25 , 2012 . Archived from the original on July 9 , 2012 . Jump up ^ `` 2012 Best Cars
for Families Awards '' . U.S. News Rankings and Reviews . Retrieved 2012 - 04 - 01 . External links ( edit ) Look up fit in Wiktionary , the free dictionary .
Wikimedia Commons has media related to Honda Fit , Honda Jazz and Honda Fit EV . Fit / Jazz at Honda Worldwide Honda Electric Vehicles Honda Fit Hybrid video at
Honda World site Honda recalls since 1977 Honda Motor Company Divisions and subsidiaries Acura American Honda Honda Canada Canada Manufacturing Aircraft
Performance Development India Motorcycles Racing F1 Taiwan China ( 65 % ) UK Joint ventures and shareholdings Bangladesh Honda Private Limited ( 70 % ) Dongfeng
Honda ( 50 % ) GE Honda Aero
</item 8>

<item 9>
Honda Fit
( April 28 , 2014 ) . `` Smaller , Larger '' . Autoweek. 64 ( 9 ) : 28 . Jump up ^ `` Honda Jazz 2014 specifications ( Thailand ) '' ( PDF ) . Jump up ^ `` 2015
US - spec Honda Fit launched in Detroit '' . worldcarfans . Retrieved 2014 - 04 - 16 . ^ Jump up to : `` New Honda Fit introduced with `` Earth Dreams '' engine
, CVT , new body construction `` . sae.org . Jump up ^ `` Honda Fit specs in Japan '' . Retrieved 2014 - 04 - 16 . Jump up ^ `` Honda Worldwide - September 5 ,
2013 `` Honda to Release all - New Fit and Fit Hybrid in Japan '' `` . honda.com . Archived from the original on December 2 , 2013 . Jump up ^
http://www.extremetech.com/extreme/174741-2015-honda-fit-brings-more-tech-to-the-subcompact-market-navigation-for-60-anyone The 2015 Fit was manufactured in
Honda 's new Mexico plant .
</item 9>

<item 10>
Honda Fit
Jump up ^ `` 2008 Honda Fit Specs '' . JB car pages . Retrieved 2009 - 03 - 09 . ^ Jump up to : `` 2009 Honda Fit Specs '' . JB car pages . Retrieved 2009 - 03
- 09 . ^ Jump up to : Lyon , Peter ( 2007 - 09 - 20 ) . `` First Drive : 2009 Honda Fit -- The Big Success Gets Bigger '' . Edmunds . Archived from the original
on September 26 , 2007 . Retrieved 2007 - 09 - 23 . Jump up ^ Kubota , Yoko ( Sep 21 , 2012 ) . `` Honda aims to nearly double car sales in four years '' .
Reuters . Retrieved 19 February 2017 . Jump up ^ Makiko Kitamura and Yuki Hagiwa ( 2010 - 10 - 08 ) . `` Honda Unveils Fit Hybrid , Considers Overseas Output ''
. Bloomberg .
</item 10>
=== END ITEM id=991a72b506ea6782 ===

=== ITEM id=c6f50181fc6f723b ===
Gold sentences and facts:
Sentence 0: It was suggested that the interior of the 45 metre ( 147 ft ) wide dome should be covered with a mosaic decoration to make the most of the available light coming through the circular windows of the drum and through the lantern . Brunelleschi had proposed the vault to glimmer with resplendent gold , but his death in 1446 put an end to this project , and the walls of the dome were whitewashed . Grand Duke Cosimo I de ' Medici decided to have the dome painted with a representation of The Last Judgment . This enormous work , 3,600 metres2 ( 38 750 ft2 ) of painted surface , was started in 1568 by Giorgio Vasari and Federico Zuccari and would last till 1579 . The upper portion , near the lantern , representing The 24 Elders of Apoc. 4 was finished by Vasari before his death in 1574 . Federico Zuccari and a number of collaborators , such as Domenico Cresti , finished the other portions : ( from top to bottom ) Choirs of Angels ; Christ , Mary and Saints ; Virtues , Gifts of the Holy Spirit and Beatitudes ; and at the bottom of the cupola : Capital Sins and Hell . These frescoes are considered Zuccari 's greatest work . But the quality of the work is uneven because of the input of different artists and the different techniques . Vasari had used true fresco , while Zuccari had painted in secco . During the restoration work , which ended in 1995 , the entire pictorial cycle of The Last Judgment was photographed with specially designed equipment and all the information collected in a catalogue . All the restoration information along with reconstructed images of the frescos were stored and managed in the Thesaurus Florentinus computer system .
  Fact 0.0: It was suggested that the interior of the 45 metre (147 ft) wide dome should be covered with a mosaic decoration.
  Fact 0.1: The mosaic decoration was suggested to make the most of the available light coming through the circular windows of the drum.
  Fact 0.2: The mosaic decoration was suggested to make the most of the available light coming through the lantern.
  Fact 0.3: Brunelleschi had proposed that the vault glimmer with resplendent gold.
  Fact 0.4: Brunelleschi died in 1446.
  Fact 0.5: Brunelleschi's death in 1446 put an end to the gold vault project.
  Fact 0.6: The walls of the dome were whitewashed.
  Fact 0.7: Grand Duke Cosimo I de' Medici decided to have the dome painted with a representation of The Last Judgment.
  Fact 0.8: The painting of The Last Judgment covers 3,600 square metres (38,750 square feet) of painted surface.
  Fact 0.9: The painting of The Last Judgment was started in 1568 by Giorgio Vasari and Federico Zuccari.
  Fact 0.10: The painting of The Last Judgment would last until 1579.
  Fact 0.11: The upper portion of the painting, near the lantern, represents The 24 Elders of Apoc. 4.
  Fact 0.12: Vasari finished the upper portion of the painting before his death in 1574.
  Fact 0.13: Federico Zuccari and a number of collaborators, such as Domenico Cresti, finished the other portions of the painting.
  Fact 0.14: The other portions, from top to bottom, are Choirs of Angels; Christ, Mary and Saints; Virtues, Gifts of the Holy Spirit and Beatitudes; and at the bottom of the cupola, Capital Sins and Hell.
  Fact 0.15: The frescoes are considered Zuccari's greatest work.
  Fact 0.16: The quality of the work is uneven because of the input of different artists and the different techniques.
  Fact 0.17: Vasari had used true fresco.
  Fact 0.18: Zuccari had painted in secco.
  Fact 0.19: The restoration work ended in 1995.
  Fact 0.20: During the restoration work, the entire pictorial cycle of The Last Judgment was photographed with specially designed equipment.
  Fact 0.21: All the information collected during the restoration was put into a catalogue.
  Fact 0.22: All the restoration information, along with reconstructed images of the frescos, was stored and managed in the Thesaurus Florentinus computer system.

Retrieved items:
<item 1>
Florence Cathedral
Jump up ^ As referenced in `` Cupola di Santa Maria del Fiore : il cantiere di restauro 1980 - 1995 '' , a cura di Cristina Acidini Luchinat e Riccardo Dalla
Negra published by Istituto Poligrafico e Zecca dello Stato ( Roma ) in 1995 ( ISBN 8824039561 ) Jump up ^ Thesaurus Florentinus project page ( in Italian ) ,
Soprintendenza ai Beni Architettonici e Paesaggisitici di Firenze , Ministero dei Beni Culturali References ( edit ) Bartlett , Kenneth R. ( 1992 ) . The
Civilization of the Italian Renaissance . Toronto : D.C. Heath and Company . ISBN 0 - 669 - 20900 - 7 ( Paperback ) . King , Ross ( 2000 ) . Brunelleschi 's
Dome : How a Renaissance Genius Reinvented Architecture . London : Chatto & Windus . Jepson , Tim ( 2001 ) . The National Geographic Traveler , Florence &
Tuscany . National Geographic Society .
</item 1>

<item 2>
Florence Cathedral
This timetable was used until the 18th century . This is one of the few clocks from that time that still exist and are in working order . The church is
particularly notable for its 44 stained glass windows , the largest undertaking of this kind in Italy in the 14th and 15th century . The windows in the aisles
and in the transept depict saints from the Old and the New Testament , while the circular windows in the drum of the dome or above the entrance depict Christ
and Mary . They are the work of the greatest Florentine artists of their times , such as Donatello , Lorenzo Ghiberti , Paolo Uccello and Andrea del Castagno .
Christ crowning Mary as Queen , the stained - glass circular window above the clock , with a rich range of coloring , was designed by Gaddo Gaddi in the early
14th century . Donatello designed the stained - glass window ( Coronation of the Virgin ) in the drum of the dome ( the only one that can be seen from the nave
) .
</item 2>

<item 3>
Florence Cathedral
He was declared the winner over his competitors Lorenzo Ghiberti and Antonio Ciaccheri . His design ( now on display in the Museum Opera del Duomo ) was for an
octagonal lantern with eight radiating buttresses and eight high arched windows . Construction of the lantern was begun a few months before his death in 1446 .
Then , for 15 years , little progress was possible , due to alterations by several architects . The lantern was finally completed by Brunelleschi 's friend
Michelozzo in 1461 . The conical roof was crowned with a gilt copper ball and cross , containing holy relics , by Verrocchio in 1469 . This brings the total
height of the dome and lantern to 114.5 meters ( 375 ft ) . This copper ball was struck by lightning on 17 July 1600 and fell down . It was replaced by an even
larger one two years later . Cupola of the Dome The commission for this bronze ball ( atop the lantern ) went to the sculptor Andrea del Verrocchio , in whose
workshop there was at this time a young apprentice named Leonardo da Vinci .
</item 3>

<item 4>
Florence Cathedral
Vasari had used true fresco , while Zuccari had painted in secco . During the restoration work , which ended in 1995 , the entire pictorial cycle of The Last
Judgment was photographed with specially designed equipment and all the information collected in a catalogue . All the restoration information along with
reconstructed images of the frescos were stored and managed in the Thesaurus Florentinus computer system . Crypt ( edit ) Tomb of Filippo Brunelleschi . The
cathedral underwent difficult excavations between 1965 and 1974 . The archaeological history of this huge area was reconstructed through the work of Dr.
Franklin Toker : remains of Roman houses , an early Christian pavement , ruins of the former cathedral of Santa Reparata and successive enlargements of this
church . Close to the entrance , in the part of the crypt open to the public , is the tomb of Brunelleschi . While its location is prominent , the actual tomb
is simple and humble . That the architect was permitted such a prestigious burial place is proof of the high esteem he was given by the Florentines .
</item 4>

<item 5>
Florence Cathedral
Above this shrine is the painting Last Supper by the lesser - known Giovanni Balducci . There was also a glass - paste mosaic panel The Bust of Saint Zanobius
by the 16th - century miniaturist Monte di Giovanni , but it is now on display in the Museum Opera del Duomo . Many decorations date from the 16th - century
patronage of the Grand Dukes , such as the pavement in colored marble , attributed to Baccio d'Agnolo and Francesco da Sangallo ( 1520 -- 26 ) . Some pieces of
marble from the façade were used , topside down , in the flooring ( as was shown by the restoration of the floor after the 1966 flooding ) . It was suggested
that the interior of the 45 metre ( 147 ft ) wide dome should be covered with a mosaic decoration to make the most of the available light coming through the
circular windows of the drum and through the lantern . Brunelleschi had proposed the vault to glimmer with resplendent gold , but his death in 1446 put an end
to this project , and the walls of the dome were whitewashed .
</item 5>

<item 6>
Florence Cathedral
This fresco , transferred on canvas in the 19th century , in the same style as the previous one , is painted in a color resembling marble . However , it is more
richly decorated and gives more the impression of movement . Both frescoes portray the condottieri as heroic figures riding triumphantly . Both painters had
problems when applying in painting the new rules of perspective to foreshortening : they used two unifying points , one for the horse and one for the pedestal ,
instead a single unifying point . Busts of Giotto ( by Benedetto da Maiano ) , Brunelleschi ( by Buggiano - 1447 ) , Marsilio Ficino , and Antonio Squarcialupi
( a most famous organist ) . These busts all date from the 15th and the 16th centuries . Above the main door is the colossal clock face with fresco portraits of
four Prophets or Evangelists by Paolo Uccello ( 1443 ) . This one - handed liturgical clock shows the 24 hours of the hora italica ( Italian time ) , a period
of time ending with sunset at 24 hours .
</item 6>

<item 7>
Florence Cathedral
As this cathedral was built with funds from the public , some important works of art in this church honour illustrious men and military leaders of Florence :
Lorenzo Ghiberti had a large artistic impact on the cathedral . Ghiberti worked with Filippo Brunelleschi on the cathedral for eighteen years and had a large
number of projects on almost the whole east end . Some of his works were the stained glass designs , the bronze shrine of Saint Zenobius and marble revetments
on the outside of the cathedral . Dante Before the City of Florence by Domenico di Michelino ( 1465 ) . This painting is especially interesting because it shows
us , apart from scenes of the Divine Comedy , a view on Florence in 1465 , a Florence such as Dante himself could not have seen in his time . Funerary Monument
to Sir John Hawkwood by Paolo Uccello ( 1436 ) . This almost monochrome fresco , transferred to canvas in the 19th century , is painted in terra verde , a color
closest to the patina of bronze . Equestrian statue of Niccolò da Tolentino by Andrea del Castagno ( 1456 ) .
</item 7>

<item 8>
Florence Cathedral
Grand Duke Cosimo I de ' Medici decided to have the dome painted with a representation of The Last Judgment . This enormous work , 3,600 metres2 ( 38 750 ft2 )
of painted surface , was started in 1568 by Giorgio Vasari and Federico Zuccari and would last till 1579 . The upper portion , near the lantern , representing
The 24 Elders of Apoc. 4 was finished by Vasari before his death in 1574 . Federico Zuccari and a number of collaborators , such as Domenico Cresti , finished
the other portions : ( from top to bottom ) Choirs of Angels ; Christ , Mary and Saints ; Virtues , Gifts of the Holy Spirit and Beatitudes ; and at the bottom
of the cupola : Capital Sins and Hell . These frescoes are considered Zuccari 's greatest work . But the quality of the work is uneven because of the input of
different artists and the different techniques . Vasari had used true fresco , while Zuccari had painted in secco .
</item 8>

<item 9>
Florence Cathedral
Unsourced material may be challenged and removed . ( July 2012 ) ( Learn how and when to remove this template message ) Interior of the cathedral Huge clock
decorated by Paolo Uccello . Dante and the Divine Comedy . Trompe l'oeil of Niccolò da Tolentino . Vasari 's fresco begun in 1568 , and completed by Federico
Zuccari in 1579 . Detail under the dome . The Last Judgement ( detail ) under the dome . Tomb of Antonio d'Orso by Tino da Camaino . The Gothic interior is vast
and gives an empty impression . The relative bareness of the church corresponds with the austerity of religious life , as preached by Girolamo Savonarola . Many
decorations in the church have been lost in the course of time , or have been transferred to the Museum Opera del Duomo , such as the magnificent cantorial
pulpits ( the singing galleries for the choristers ) of Luca della Robbia and Donatello .
</item 9>

<item 10>
Florence Cathedral
The two main competitors were two master goldsmiths , Lorenzo Ghiberti and Filippo Brunelleschi , the latter of whom was supported by Cosimo de Medici .
Ghiberti had been the winner of a competition for a pair of bronze doors for the Baptistery in 1401 and lifelong competition between the two remained sharp .
Brunelleschi won and received the commission . Ghiberti , appointed coadjutator , drew a salary equal to Brunelleschi 's and , though neither was awarded the
announced prize of 200 florins , was promised equal credit , although he spent most of his time on other projects . When Brunelleschi became ill , or feigned
illness , the project was briefly in the hands of Ghiberti . But Ghiberti soon had to admit that the whole project was beyond him . In 1423 , Brunelleschi was
back in charge and took over sole responsibility . Work started on the dome in 1420 and was completed in 1436 . The cathedral was consecrated by Pope Eugene IV
on March 25 , 1436 , ( the first day of the year according to the Florentine calendar ) .
</item 10>
=== END ITEM id=c6f50181fc6f723b ===

=== ITEM id=1fc15aa392406485 ===
Gold sentences and facts:
Sentence 0: Songwriter ( s ) Kirsty MacColl / Philip Rambow
  Fact 0.0: The songwriters of the song are Kirsty MacColl and Philip Rambow.
Sentence 1: `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 )
  Fact 1.0: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is a single by Kirsty MacColl.
  Fact 1.1: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is from the album Desperate Character.
  Fact 1.2: The single's A-side is 'There's a Guy Works Down the Chip Shop Swears He's Elvis'.
  Fact 1.3: The single's B-side is 'Hard To Believe There's a Guy ... (country version)'.
  Fact 1.4: The single was released in May 1981.
  Fact 1.5: The single's format is 7-inch vinyl.
  Fact 1.6: The single's genre is country pop.
  Fact 1.7: The single's label is Polydor.
  Fact 1.8: The single's songwriters are Kirsty MacColl and Philip Rambow.
  Fact 1.9: The single's producer is Barry Farmer.
  Fact 1.10: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981).
  Fact 1.11: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' (1981).
  Fact 1.12: The Kirsty MacColl singles chronology includes 'See That Girl' (1981).
  Fact 1.13: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981) a second time.
  Fact 1.14: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop Swears He's Elvis' (1981).
  Fact 1.15: The Kirsty MacColl singles chronology includes 'See That Girl' (1981) a second time.

Retrieved items:
<item 1>
There's a Guy Works Down the Chip Shop Swears He's Elvis
There 's a Guy Works Down the Chip Shop Swears He 's Elvis - wikipedia There 's a Guy Works Down the Chip Shop Swears He 's Elvis Jump to : navigation , search
This article relies too much on references to primary sources . Please improve this by adding secondary or tertiary sources . ( March 2008 ) ( Learn how and
when to remove this template message ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate
Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released
May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles
chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop ,
</item 1>

<item 2>
Lord Have Mercy on the Working Man
Critical reception ( edit ) Teresa M. Walker , in her review for the Gainesville Sun , said that with the assistance from superstars on the final chorus , the
song `` should shoot up the charts . '' Dave Larsen of the Dayton Daily News cited it as one of the stronger tracks on the album , saying that the album ``
works best when Tritt sticks with the populist approach . '' Alanna Nash of Entertainment Weekly said of the song , `` Tritt finds a solid image for his laborer
's lament ( ... ) But the melody is so slight that he resorts to an acoustic arrangement that evokes Jimmie Rodgers and the young Roy Acuff . '' Deborah Evans
Price , of Billboard magazine gave the song a mixed review , saying that while the final chorus of song features famous country artists , it was too bad that
the entire song is n't as strong as the last verse . Music video ( edit ) Jack Cole directed the song 's music video .
</item 2>

<item 3>
Neil Diamond
He did songwriting wherever he could , including on buses , and used an upright piano above the Birdland Club in New York City . One of the causes of this early
nomadic life as a songwriter was due to his songs having too many words : `` I 'd spent a lot of time on lyrics , and they were looking for hooks , and I did
n't really understand the nature of that , '' he says . During those lean years , he was only able to sell about one song a week , barely enough to survive on .
He found himself only earning enough to spend 35 cents a day on food ( US $3 in 2017 dollars ) . However , the privacy he had above the Birdland Club allowed
him to focus on writing without distractions ; as he explained , `` Something new began to happen . I was n't under the gun , and suddenly interesting songs
began to happen , songs that had things none of the others did . '' Among them were `` Cherry , Cherry '' and `` Solitary Man '' . `` Solitary Man '' was the
first record that Diamond recorded in his own name that made the charts .
</item 3>

<item 4>
All I Want for Christmas Is My Two Front Teeth
The song was introduced in 1948 by singing group The Satisfiers on Perry Como 's radio show , and originally recorded by Spike Jones & His City Slickers on
December 6 , 1947 , with lead vocal by George Rock . That version reached the top of the pop charts in 1948 , and again in 1949 . The song has also been
recorded by numerous other singers and performers , including Ray Stevens , George Strait , Danny Kaye with The Andrews Sisters , Urbie Green , The Platters ,
Dread Zeppelin , The Kelly Family , Nat King Cole ( reportedly Gardner 's favorite version ) , The Chipmunks , the Hampton String Quartet , The Three Stooges
and Count von Count of Sesame Street . In the versions by Spike , The Chipmunks ( Theodore on lead vocals ) , and Ray Stevens , the song ends with the performer
shouting the words `` Happy New Year ! ! ! '' Homer and Jethro parodied the song as `` All I Want for Christmas Is My Upper Plate '' , on their 1968 album Cool
Crazy Christmas .
</item 4>

<item 5>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Retrieved 2014 - 10 - 30 . Jump up ^ Information at Svensk mediedatabas External links ( edit ) freeworld - Kirsty MacColl fansite Kirsty MacColl Studio albums
Desperate Character Kirsty MacColl Kite Electric Landlady Titanic Days Tropical Brainstorm Compilations Galore Notable singles `` They Do n't Know '' `` Keep
Your Hands Off My Baby '' `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' `` You Still Believe in Me '' `` Terry '' `` A New England '' `` Days
'' `` Do n't Come The Cowboy With Me Sonny Jim ! ''
</item 5>

<item 6>
Let My Love Open the Door
Chris Thomas Pete Townshend UK singles chronology `` Rough Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) `` Rough
Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) Pete Townshend US singles chronology `` Keep Me Turning '' ( 1977 ) (
with Ronnie Lane ) Keep Me Turning1977 `` Let My Love Open the Door '' ( 1980 ) Let My Love Open the Door1980 `` A Little Is Enough '' ( 1980 ) A Little Is
Enough 1980 `` Let My Love Open the Door '' is a song written and performed by Pete Townshend from his 1980 album Empty Glass . It reached the top ten in the
United States in that same year , reaching number nine . It reached number five in Canada .
</item 6>

<item 7>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Track listing ( edit ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( K. MacColl / P. Rambow ) `` Hard to Believe '' ( K. MacColl ) `` There
's a Guy Works Down the Chip Shop Swears He 's Elvis ( country version ) '' ( K. MacColl / P. Rambow ) References ( edit ) Jump up ^ `` There 's a Guy Works
Down the Chip Shop Swears He 's Elvis - Kirsty MacColl '' . Retrieved 2014 - 10 - 30 . Jump up ^ `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis 7
'' single - Kirsty MacColl `` . Retrieved 2014 - 10 - 30 .
</item 7>

<item 8>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's
Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' was Kirsty MacColl 's first hit single ,
charting in the UK at # 14 in 1981 , and remaining in the charts for nine weeks . It reached # 9 in the Irish charts . A promotional version was released in the
USA , which instead referenced a truck stop as opposed to the very British chip shop , but this was only available to DJs and did not get a full release .
Norwegian singer Elisabeth Andreasson covered the song on her 1981 country album Angel of the Morning , with lyrics in Swedish by Hasse Olsson as `` Killen ner
' på Konsum svär att han är Elvis '' ( `` The guy down Konsum '' swears he 's Elvis ) .
</item 8>

<item 9>
A Whiter Shade of Pale
You 're a Rich Man '' by The Beatles Procol Harum Gary Brooker Geoff Dunn Matt Pegg Josh Phillips Geoff Whitehorn Dave Ball Graham Broad Dave Bronze Pete Brown
Mark Brzezicki Alan Cartwright Chris Copping Matthew Fisher Mick Grabham Bobby Harrison David Knights Dee Murray Keith Reid Tim Renwick Ray Royer Don Snow Peter
Solley Henry Spinetti Jerry Stevenson Robin Trower Ian Wallace B.J.
</item 9>

<item 10>
There's a Guy Works Down the Chip Shop Swears He's Elvis
`` Walking Down Madison '' `` Soho Square '' With The Pogues `` Fairytale of New York '' Related articles Ewan MacColl Steve Lillywhite Retrieved from ``
https://en.wikipedia.org/w/index.php?title=There%27s_a_Guy_Works_Down_the_Chip_Shop_Swears_He%27s_Elvis&oldid=799608311 '' Categories : 1981 singles 1981 songs
Elisabeth Andreassen songs Kirsty MacColl songs Songs written by Kirsty MacColl Songs about Elvis Presley Polydor Records singles Hidden categories : Articles
lacking reliable references from March 2008 All articles lacking reliable references Talk About Wikipedia Svenska Edit links This page was last edited on 8
September 2017 , at 19 : 12 . About Wikipedia
</item 10>
=== END ITEM id=1fc15aa392406485 ===

=== ITEM id=2ae11458e7658336 ===
Gold sentences and facts:
Sentence 0: `` Fore ! '' , originally an Australian interjection , is used to warn anyone standing or moving in the flight of a golf ball . The mention of the term in an 1881 Australian Golf Museum indicates that the term was in use at least as early as that period .
  Fact 0.0: Fore is originally an Australian interjection.
  Fact 0.1: Fore is used to warn anyone standing or moving in the flight of a golf ball.
  Fact 0.2: An 1881 Australian Golf Museum mention of the term exists.
  Fact 0.3: The 1881 mention indicates that the term Fore was in use at least as early as 1881.

Retrieved items:
<item 1>
Fore (golf)
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Fore_(golf)&oldid=802796028 '' Categories : Golf
terminology Scottish English Golf stubs Hidden categories : All articles with specifically marked weasel - worded phrases Articles with specifically marked
weasel - worded phrases from January 2016 All stub articles Talk About Wikipedia Hrvatski Italiano Norsk Srpskohrvatski / српскохрватски Svenska Edit links This
page was last edited on 28 September 2017 , at 14 : 56 . About Wikipedia
</item 1>

<item 2>
Fore (golf)
Fore ( golf ) - Wikipedia Fore ( golf ) Jump to : navigation , search The tough rubber core of a golf ball makes it a hazard to others following a wayward shot
, despite its weight not exceeding 45.93 g ( 1.62 oz ) `` Fore ! '' , originally an Australian interjection , is used to warn anyone standing or moving in the
flight of a golf ball . The mention of the term in an 1881 Australian Golf Museum indicates that the term was in use at least as early as that period . It is
believed to come from the military `` beware before '' , which an artilleryman about to fire would yell alerting nearby infantrymen to drop to the ground to
avoid the shells overhead . ( Before may mean `` in front of ( the gun being fired ) '' ; fore may mean `` ( look ) ahead '' . ) Other possible origins include
the term being derived from the term `` fore - caddie '' , a caddie waiting down range from the golfer to find where the ball lands .
</item 2>

<item 3>
Hops
Whichever was taxed made the brewer then quickly switch to the other . In Britain , hopped beer was first imported from Holland around 1400 , yet hops were
condemned as late as 1519 as a `` wicked and pernicious weed '' . In 1471 , Norwich , England , banned use of the plant in the brewing of ale ( `` beer '' was
the name for fermented malt liquors bittered with hops ; only in recent times are the words often used as synonyms ) . In Germany , using hops was also a
religious and political choice in the early 16th century . There was no tax on hops to be paid on the Catholic church , unlike on gruit , for which the
Protestant preferred hopped beer . Hops used in England were imported from France , Holland and Germany with import duty paid for those ; it was not until 1524
that hops were first grown in the southeast of England ( Kent ) when they were introduced as an agricultural crop by Dutch farmers . Therefore , in the hop
industry there are many words which originally were Dutch words ( see oast house ) . Hops were then grown as far north as Aberdeen , near breweries for
infrastructure convenience .
</item 3>

<item 4>
Pencil
Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people
have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the
element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally
mean lead pen . The value of graphite would soon be realised to be enormous , mainly because it could be used to line the moulds for cannonballs ; the mines
were taken over by the Crown and were guarded . When sufficient stores of graphite had been accumulated , the mines were flooded to prevent theft until more was
required . The usefulness of graphite for pencils was discovered as well , but graphite for pencils had to be smuggled . The news of the usefulness of these
early pencils spread far and wide , attracting the attention of artists all over the known world . Because graphite is soft , it requires some form of
encasement .
</item 4>

<item 5>
Orc (Middle-earth)
The original edition of The Hobbit and early drafts of The Lord of the Rings first used goblin everywhere and used hobgoblin for larger , more evil goblins .
While investigating possible sources for the word `` Hobbit '' , Tolkien realised he had made a mistake in using hob - , which is traditionally used to mean a
smaller entity , not a larger one . In later writings , after The Lord of the Rings , ( including The Adventures of Tom Bombadil and many essays published in
The Peoples of Middle - earth ) , Tolkien preferred the spelling Ork , evidently mainly to avoid the form Orcish , which would be naturally pronounced with the
c as / s / instead of / k / in English . Tolkien indeed used the adjective Orkish . According to George MacDonald Fraser in The Light 's on at Signpost ,
Tolkien answered a letter on this subject , confirming that Orcs and goblins were indeed identical . Physical appearance ( edit ) Orcs are described as ugly and
filthy fanged humanoids .
</item 5>

<item 6>
Fore (golf)
These caddies were often warned about oncoming golf balls by a shout of the term `` fore - caddie '' which was eventually shortened to just `` fore ! '' . The
Colonel Bogey March is based on the descending minor third which the original Colonel Bogey whistled instead of yelling Fore around 1914 . Notes ( edit ) Jump
up ^ SND : Fore ^ Jump up to : Why Do Golfers Yell `` Fore '' for Errant Shots ? retrieved June 15 , 2007 Jump up ^ Derivation 4 Fore ! and Caddie retrieved
June 15 , 2007 Jump up ^ The Windsor Magazine : An Illustrated Monthly for Men and Women Vol. 33 , no . 192 ( December 1910 ) References ( edit ) Scott , James
Sibbald David , The British Army : Its Origin , Progress , and Equipment , 1868 Windsor Magazine : An Illustrated Monthly for Men and Women Vol. 32 , no . 292 (
December 1910 ) This golf - related article is a stub . You can help Wikipedia by expanding it .
</item 6>

<item 7>
Graph query row:
entities: ['Fore!', 'beware before']
relationships: ['RELATED_TO']
historical_evidence: ['believed to come from']
</item 7>

<item 8>
Orc (Middle-earth)
About the goblins of The Hobbit , Tolkien wrote : They are not based on direct experience of mine ; but owe , I suppose , a good deal to the goblin tradition
... especially as it appears in George MacDonald , except for the soft feet which I never believed in . Middle - earth terms for Orcs ( edit ) In the High -
elven tongue Quenya , the word for `` Orc '' was urco , plural urqui , meaning `` bogey '' , or `` bogeyman '' , that is , something that provokes fear . In the
Grey - elven tongue Sindarin , it was orch , plural yrch . In the Dwarven tongue Khuzdul , it was rukhs , plural rakhâs . In the language of the Drúedain or
Wild Men , it was gorgûn . In the Black Speech of Mordor , the equivalent was Uruk , as can be seen in Uruk - hai , `` Orc - folk '' . Orc itself is from
Rohirric and the Hobbit - language , which shared linguistic roots , but the term is clearly related to the older Elvish words .
</item 8>

<item 9>
Graph query row:
entities: ['Fore!', 'Australian Golf Museum', None, 'beware before']
relationships: ['RELATED_TO', 'FROM_CHUNK', 'FROM_CHUNK']
historical_evidence: ['mentioned in 1881', None, None]
</item 9>

<item 10>
God Save the Queen
The phrase `` God Save the King '' is much older than the song , appearing , for instance , several times in the King James Bible . A text based on the 1st Book
of Kings Chapter 1 : verses 38 -- 40 , `` ... And all the people rejoic 'd , and said : God save the King ! Long live the King ! May the King live for ever ,
Amen '' , has been sung at every coronation since that of King Edgar in 973 . Scholes says that as early as 1545 `` God Save the King '' was a watchword of the
Royal Navy , with the response being `` Long to reign over us '' . He also notes that the prayer read in churches on anniversaries of the Gunpowder Plot
includes words which might have formed part of the basis for the second verse `` Scatter our enemies ... assuage their malice and confound their devices '' . In
1745 , The Gentleman 's Magazine published `` God save our lord the king : A new song set for two voices '' , describing it `` As sung at both Playhouses '' (
the Theatres Royal at Drury Lane and Covent Garden ) .
</item 10>
=== END ITEM id=2ae11458e7658336 ===
