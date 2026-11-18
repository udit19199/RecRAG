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

=== ITEM id=1428c0d9d2575aa7 ===
Gold sentences and facts:
Sentence 0: Reading Full name Reading Football Club Nickname ( s ) The Royals Founded 1871 ; 147 years ago ( 1871 ) Ground Madejski Stadium Ground Capacity 24,161 Owner Dai Yongge and Dai Xiuli ( majority ) Chairman Sir John Madejski Manager Jaap Stam League Championship 2016 -- 17 Championship , 3rd Website Club website Home colours Away colours Current season
  Fact 0.0: The full name of the club is Reading Football Club.
  Fact 0.1: The club's nickname is The Royals.
  Fact 0.2: Reading was founded in 1871.
  Fact 0.3: Reading was founded 147 years ago.
  Fact 0.4: Reading's ground is Madejski Stadium, with a ground capacity of 24,161.
  Fact 0.5: Reading's owners are Dai Yongge and Dai Xiuli (majority).
  Fact 0.6: Reading's chairman is Sir John Madejski.
  Fact 0.7: Reading's manager is Jaap Stam.
  Fact 0.8: Reading competed in the Championship in the 2016-17 season, finishing 3rd.
  Fact 0.9: Reading's website is listed as the club website.
Sentence 1: Reading Football Club ownership structure : 75 % Owned by Renhe Sports Management Ltd , 100 % owned by Xiu Li Dai and Yongge Dai . 25 % Owned by Narin Niruttinanon
  Fact 1.0: Reading Football Club's ownership structure is 75% owned by Renhe Sports Management Ltd.
  Fact 1.1: Renhe Sports Management Ltd is 100% owned by Xiu Li Dai and Yongge Dai.
  Fact 1.2: 25% of Reading Football Club is owned by Narin Niruttinanon.

Retrieved items:
<item 1>
Reading F.C.
Retrieved on 17 July 2013 . Jump up ^ HOW SOUTH GEORGIA UNITED BECAME A READING FC AFFILIATE CLUB . South Georgia United . Retrieved on 5 June 2014 .
Bibliography ( edit ) Digby , Bob ( 2001 ) . It 's a World Thing . Oxford , Oxfordshire : Oxford University Press . ISBN 978 - 0 - 19 - 913428 - 1 . Retrieved 2
June 2011 . Yeo , Stephen ( 1976 ) . Religion and Voluntary Organisations in Crisis . Abingdon , Oxfordshire : Taylor & Francis . ISBN 978 - 0 - 85664 - 017 - 9
. Retrieved 1 June 2011 . External links ( edit ) Wikimedia Commons has media related to Reading F.C. . Official website Supporters ' Trust At Reading Reading
F.C.
</item 1>

<item 2>
Reading F.C.
On 24 May 2007 , it was announced that planning permission had been granted to extend the stadium to a capacity of 36,900 . Reading have made plans for a new
training ground at Bearwood Golf Club to replace Hogwood park their current training facility . On 5 July 2016 , at the end of Eamonn Dolan 's funeral , Reading
announced that the North Stand would be renamed the `` Eamonn Dolan Stand '' . Support ( edit ) See also : Supporters ' Trust at Reading Supporters at a Reading
match at Elm Park in 1913 In 2001 , Reading became the first football club to register their fans as an official member of their squad , giving the `` player ''
registered with squad number 13 as ' Reading Fans ' . Rivalries ( edit ) Before going out of business in 1992 , Aldershot were Reading 's biggest rivals . There
was a strong rift between the two sets of fans , with fighting between fans occurring on several occasions . Strong feelings remain between fans of Reading and
fans of Aldershot Town , the refounded club in Aldershot .
</item 2>

<item 3>
Reading F.C.
Reading FC - Wikipedia Reading FC Jump to : navigation , search Not to be confused with Reading F.C. Women . Reading Full name Reading Football Club Nickname (
s ) The Royals Founded 1871 ; 147 years ago ( 1871 ) Ground Madejski Stadium Ground Capacity 24,161 Owner Dai Yongge and Dai Xiuli ( majority ) Chairman Sir
John Madejski Manager Jaap Stam League Championship 2016 -- 17 Championship , 3rd Website Club website Home colours Away colours Current season Reading Football
Club ( / ˈrɛdɪŋ / ( listen ) RED - ing ) is a professional association football club based in Reading , Berkshire , England . The team play in the Championship
, the second tier of English football . Reading are nicknamed The Royals , due to Reading 's location in the Royal County of Berkshire , though they were
previously known as The Biscuitmen , due to the town 's association with Huntley and Palmers .
</item 3>

<item 4>
Reading F.C.
2006 -- 07 LMA Championship Manager of the Year Brian McDermott 2011 -- 12 Affiliated clubs ( edit ) Albion Hurricanes FC ( 2010 -- present ) Boreham Wood (
2012 -- present ) Charlotte United FC ( 2007 -- present ) Ernakulam Eagles ( 2013 -- present ) Galatasaray ( 2013 -- present ) Ocean City Nor'easters ( 2008 --
present ) South Georgia United ( 2008 -- present ) References ( edit ) Jump up ^ `` Madejski Stadium information '' . readingfc.co.uk . Retrieved 14 April 2011
. ^ Jump up to : `` The History of Reading Football Club '' . readingfc.co.uk. 2 May 2010 . Archived from the original on 4 February 2010 . Retrieved 29
November 2010 . Jump up ^ Discover gems of our heritage Archived 7 March 2012 at the Wayback Machine. , Reading Chronicle , 10 September 2009 . Jump up ^ `` Elm
Park Reading '' . Old Football Grounds .
</item 4>

<item 5>
Reading F.C.
Reading were promoted to English football 's top division for the first time in their history . The 2006 -- 07 season saw Reading make their first appearance in
the top flight of English football . Reading defied pre-season predictions of relegation to finish the season in eighth place with 55 points . Reading turned
down the chance to play in the UEFA Intertoto Cup . In the run up to their second season in the Premier League , Reading took part in the 2007 Peace Cup in
South Korea . This second season was less successful , however , and Reading were relegated back to the Championship . Reading started the 2008 -- 09 season
with a 15 match unbeaten home run . They finished fourth and qualify for the play - offs , where they lost to Burnley in the semi-final . Manager Steve Coppell
resigned just hours after the game , replaced by Brendan Rodgers . Rodgers left the club by mutual consent on 16 December 2009 and Brian McDermott made
caretaker manager the same day .
</item 5>

<item 6>
Reading F.C.
25 % Owned by Narin Niruttinanon Reading F.C. Directors : Xiu Li Dai , Yongge Dai , Ronald Gourlay , Narin Niruttinanon , Nigel Howe Players ( edit ) As of 8
January 2018 First - team squad ( edit ) Note : Flags indicate national team as defined under FIFA eligibility rules . Players may hold more than one non-FIFA
nationality . No . Position Player GK Vito Mannone DF Chris Gunter MF Joey van den Berg 5 DF Paul McShane ( captain ) 6 DF Liam Moore 7 MF Roy Beerens 8 MF
George Evans 9 FW Joseph Mendes 10 MF John Swift 11 DF Jordan Obita 12 MF Garath McCleary 14 FW Sone Aluko 15 MF Callum Harriott 16 MF David Edwards 17 FW Modou
Barrow 18 FW Yann Kermorgant 19 MF Leandro Bacuna No .
</item 6>

<item 7>
Reading F.C.
. Official website Supporters ' Trust At Reading Reading F.C. on BBC Sport : Club news -- Recent results -- Upcoming fixtures Reading Football Club Current
season Under - 21s & Academy Records & Statistics Honours History History Seasons Thames Valley Royals proposal Grounds Elm Park ( 1896 -- 1998 ) Madejski
Stadium ( 1998 -- present ) People Managers Players with 1 -- 24 appearances 25 -- 99 appearances 100 + appearances Internationals Player of the Season Notable
matches Level 2 play - off finals 1995 2011 2017 Level 3 play - off finals Full Members ' Cup finals 1988 Other matches Portsmouth 7 -- 4 Reading Related
articles Hob Nob Anyone ? Supporters ' Trust At Reading Reading Women Marussia F1 Book : Reading F.C. Category : Reading F.C. Portal : Association football
Commons : Reading F.C. Reading F.C. seasons 1940 -- 41 1941 -- 42 1942 -- 43 1943 -- 44 1944 -- 45 1945 -- 46 1946 -- 47 1947 -- 48 1948 -- 49
</item 7>

<item 8>
Reading F.C.
internationals Former players ( edit ) Main article : List of Reading F.C. players Club officials ( edit ) Board of Directors & Senior Club Staff Chief
Executive : Ron Gourlay Commercial Director : Adam Benson Club Secretary : Sue Hewett Financial Controller : Bryan Stabler Directors : Sir John Madejski ( Co
Chairman ) , Sasima Srivikorn ( Co Chairwoman ) , Nigel Howe , Narin Niruttinanon , Sumrith Thanakarnjanasuth , Taweesuk Srisumrid and Theekharoj Piamphongsarn
Management team Director of Football : Brian Tevreden First Team Manager : Jaap Stam Assistant Manager : Andries Ulderink & Said Bakkati First Team Coach :
Michael Gilkes Goalkeeping Coach : Dave Beasant First Team Sports Scientist : Tristan Baker First Team Physio : Jon Urwin Academy Manager : Lee Herron Assistant
Academy Manager : David Dodds Academy Loan Manager : Mikele Leigertwood Kit Manager : Stewart Bannister First Team Head Scout : Steve Shorey Masseur : Dan
Buchanan Managers ( edit ) Main article : List of Reading F.C.
</item 8>

<item 9>
Reading F.C.
Reading then finished eighth in the 2006 -- 07 Premier League , their first ever season as a top flight club . Contents ( hide ) 1 History 2 Crest and colours 3
Stadium 4 Support 4.1 Rivalries 5 Sponsorship 5.1 Additional kit sponsors 6 Ownership and finances 7 Players 7.1 First - team squad 7.2 Under - 23 squad 7.3
Under - 18 squad 7.4 Out on loan 8 Records and statistics 8.1 Notable players 8.2 Player of the season 8.3 International players 8.4 Former players 9 Club
officials 10 Managers 11 Women 's team 12 Honours 12.1 Managerial 13 Affiliated clubs 14 References 15 Bibliography 16 External links History ( edit ) Main
article : History of Reading F.C. See also : Thames Valley Royals proposal Reading were formed on 25 December 1871 , following a public meeting at the Bridge
Street Rooms organised by the future club secretary Joseph Edward Sydenham .
</item 9>

<item 10>
Reading F.C.
A board meeting the following year decided that the move would not be possible , as `` there was no chance of a move to the ground near to the GWR railway
stations due to the actions of the Great Western Railway '' . The Madejski Stadium has been Reading 's home ground since 1998 In 1994 , the Taylor Report made
an all - seater stadium compulsory in the top two divisions ( the Premier League and the first division ) . Reading were champions of the second division in
1994 , and were promoted to the first division . Reading became subject to the Taylor requirements , though converting Elm Park to an all - seater stadium would
have been impractical . Instead , a location in Smallmead ( to the south of the town ) was identified as the site for a new stadium . The former council
landfill site was bought for £ 1 , with further conditions that the development of the stadium would include part - funding of the A33 relief road . Expansion
of the club 's home would also allow alternative commercial ventures ( particularly leisure facilities ) and shared use with other teams ( such as rugby union
clubs Richmond and London Irish ) .
</item 10>
=== END ITEM id=1428c0d9d2575aa7 ===

=== ITEM id=7012a47a4d835481 ===
Gold sentences and facts:
Sentence 0: Group 3 in the periodic table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury ( element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson alkaline earth metals ← → group 4 IUPAC group number Name by element scandium group CAS group number ( US , pattern A-B-A ) IIIB old IUPAC number ( Europe , pattern A-B ) IIIA ↓ Period Scandium ( Sc ) 21 Transition metal 5 Yttrium ( Y ) 39 Transition metal 6 Lanthanum ( La * ) 57 Lanthanide 7 Actinium ( Ac * ) 89 Actinide * Whether the elements lutetium ( Lu ) and lawrencium ( Lr ) , in period 6 and 7 , are in group 3 is disputed . The grouping used in this article places La and Ac in group 3 , which is the most common form . For other groupings , see group 3 borders . Legend primordial element synthetic element Atomic number color : black = solid
  Fact 0.0: The periodic table has a group 3.
  Fact 0.1: The text lists the element names from Hydrogen through Oganesson.
  Fact 0.2: The text mentions alkaline earth metals.
  Fact 0.3: The article's grouping places lanthanum (La) and actinium (Ac) in group 3.
  Fact 0.4: The article says placing La and Ac in group 3 is the most common form.
  Fact 0.5: Scandium (Sc) has atomic number 21 and is a transition metal.
  Fact 0.6: Yttrium (Y) has atomic number 39, is in period 5, and is a transition metal.
  Fact 0.7: Lanthanum (La) has atomic number 57, is in period 6, and is a lanthanide.
  Fact 0.8: Actinium (Ac) has atomic number 89, is in period 7, and is an actinide.
  Fact 0.9: Whether lutetium (Lu) and lawrencium (Lr), in periods 6 and 7, are in group 3 is disputed.
  Fact 0.10: For other groupings, the article refers readers to group 3 borders.
  Fact 0.11: The scandium group has CAS group number IIIB (US pattern A-B-A).
  Fact 0.12: The scandium group has old IUPAC number IIIA (Europe pattern A-B).
  Fact 0.13: The article has a legend distinguishing primordial elements and synthetic elements.
  Fact 0.14: The article has a legend for atomic number colors, where black means solid.

Retrieved items:
<item 1>
Group 3 element
Exclusion of all elements is based on properties of earlier actinides , which show a much wider variety of chemistry ( for instance , in range of oxidation
states ) within their series than the lanthanides , and comparisons to scandium and yttrium are even less useful . However , these elements are destabilized ,
and if they were stabilized to more closely match chemistry laws , they would be similar to lanthanides as well . Also , the later actinides from californium
onwards behave more like the corresponding lanthanides , with only the valence + 3 ( and sometimes + 2 ) shown . In 2015 , IUPAC initiated a project to decide
... group 3 of the periodic table as consisting either of the elements Sc , Y , Lu and Lr , or the elements Sc , Y , La and Ac . -- IUPAC , The project is led
by Eric Scerri . Apparently , group 3 comprising 32 elements ( as older periodic tables show ) is not considered . The La and Ac variant remains the most common
in the literature , despite some calls for a change to the Lu and Lr variant .
</item 1>

<item 2>
Group 3 element
If lanthanum and actinium are considered group 3 elements , then element 121 , unbiunium ( Ubu ) , should be the fifth group 3 element . The element is
calculated have electronic configuration of ( Og ) 8s 8p , which is not associated with transition metals , without having a partially filled d - subshell . No
experiments have been performed to create unpenttrium , unbiunium or any element that could be considered the next group 3 element ; however , unbiunium is the
element with the lowest atomic number that has not been tried to be created and thus has chances to be , while unpenttrium , unpentpentium or any other element
considered if preceded by lawrencium is very unlikely to be created due to drip instabilities that imply that the periodic table ends soon after the island of
stability at unbihexium .
</item 2>

<item 3>
Group 3 element
Three group 3 elements occur naturally , scandium , yttrium , and either lanthanum or lutetium . Lanthanum continues the trend started by two lighter members in
general chemical behavior , while lutetium behaves more similarly to yttrium . While the choice of lutetium would be in accordance with the trend for period 6
transition metals to behave more similarly to their upper periodic table neighbors , the choice of lanthanum is in accordance with the trends in the s - block ,
which the group 3 elements are chemically more similar to . They all are silvery - white metals under standard conditions . The fourth element , either actinium
or lawrencium , has only radioactive isotopes . Actinium , which occurs only in trace amounts , continues the trend in chemical behavior for metals that form
tripositive ions with a noble gas configuration ; synthetic lawrencium is calculated and partially shown to be more similar to lutetium and yttrium . So far ,
no experiments have been conducted to synthesize any element that could be the next group 3 element .
</item 3>

<item 4>
Group 3 element
Characteristics ( edit ) Chemical ( edit ) Electron configurations of the group 3 elements Z Element Electron configuration 21 scandium 2 , 8 , 9 , 2 39 yttrium
2 , 8 , 18 , 9 , 2 57 lanthanum 2 , 8 , 18 , 18 , 9 , 2 89 actinium 2 , 8 , 18 , 32 , 18 , 9 , 2 Like other groups , the members of this family show patterns in
their electron configurations , especially the outermost shells , resulting in trends in chemical behavior . However , lawrencium is an exception , since its
last electron is transferred to the 7p subshell due to relativistic effects . Most of the chemistry has been observed only for the first three members of the
group ; chemical properties of both actinium and especially lawrencium are not well - characterized . The remaining elements of the group ( scandium , yttrium ,
lutetium ) are reactive metals with high melting points ( 1541 ° C , 1526 ° C , 1652 ° C respectively ) .
</item 4>

<item 5>
Group 3 element
element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium
Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium
Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson alkaline earth metals ← → group 4 IUPAC group number Name by element scandium group CAS group
number ( US , pattern A-B-A ) IIIB old IUPAC number ( Europe , pattern A-B ) IIIA ↓ Period Scandium ( Sc ) 21 Transition metal 5 Yttrium ( Y ) 39 Transition
metal 6 Lanthanum ( La * ) 57 Lanthanide 7 Actinium ( Ac * ) 89 Actinide * Whether the elements lutetium ( Lu ) and lawrencium ( Lr ) , in period 6 and 7 , are
in group 3 is disputed .
</item 5>

<item 6>
Group 3 element
in period 6 and 7 , are in group 3 is disputed . The grouping used in this article places La and Ac in group 3 , which is the most common form . For other
groupings , see group 3 borders . Legend primordial element synthetic element Atomic number color : black = solid Group 3 is a group of elements in the periodic
table . This group , like other d - block groups , should contain four elements , but it is not agreed what elements belong in the group . Scandium ( Sc ) and
yttrium ( Y ) are always included , but the other two spaces are usually occupied by lanthanum ( La ) and actinium ( Ac ) , or by lutetium ( Lu ) and lawrencium
( Lr ) ; less frequently , it is considered the group should be expanded to 32 elements ( with all the lanthanides and actinides included ) or contracted to
contain only scandium and yttrium . When the group is understood to contain all of the lanthanides , its trivial name is the rare - earth metals .
</item 6>

<item 7>
Group 3 element
So far , no experiments were conducted to synthesize any element that could be the next group 3 element ; if lutetium and lawrencium are considered to be group
3 elements , then the next element in the group should be element 153 , unpenttrium ( Upt ) . However , after element 120 , filling electronic configurations
stops obeying Aufbau principle . According to the principle , unpenttrium should have an electronic configuration of ( Og ) 8s 5g 6f 7d and filling the 5g -
subshell should be stopped at element 138 . However , the 7d - orbitals are calculated to start being filled on element 137 , while the 5g - subshell closes
only at element 144 , after filling of 7d - subshell begins . Therefore , it is hard to calculate which element should be the next group 3 element .
Calculations suggest that unpentpentium ( Upp , element 155 ) could also be the next group 3 element .
</item 7>

<item 8>
Group 3 element
^ Jump up to : Barbalace , Kenneth . `` Periodic Table of Elements '' . Environmental Chemistry.com . Retrieved 2007 - 04 - 14 . Jump up ^ `` WebElements
Periodic Table of the Elements '' . Webelements.com . Retrieved 2010 - 04 - 03 . Jump up ^ `` Periodic Table of the Elements '' . International Union of Pure
and Applied Chemistry ( IUPAC ) . Retrieved 2010 - 04 - 03 . ^ Jump up to : `` Visual Elements '' . Royal Society of Chemistry . Retrieved 4 July 2011 . Jump up
^ Dolg , Michael . `` Lanthanides and Actinides '' ( PDF ) . Max - Planck - Institut für Physik komplexer Systeme , Dresden , Germany . CLA01 . Retrieved 4 July
2011 . Jump up ^ IUPAC ( 2015 ) . `` The constitution of group 3 of the periodic table '' . Retrieved 2016 - 11 - 10 . Jump up ^ Greenwood , N.N.
</item 8>

<item 9>
Group 3 element
ISBN 0 - 19 - 850341 - 5 . Links to related articles ( edit ) ( hide ) Periodic table ( Large cells ) 5 6 7 8 9 10 11 12 13 14 15 16 17 18 He Li Be O Ne Na Mg
Al Si Cl Ar Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr 5 Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe 6 Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm
Yb Lu Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn 7 Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og Alkali metal
Alkaline earth metal Lanthanide Actinide Transition metal Post - ​ transition metal Metalloid Reactive nonmetal Noble gas Unknown chemical properties Periodic
table Periodic table forms Standard 18 - column 18 - column ,
</item 9>

<item 10>
Group 3 element
Group 3 element - wikipedia Group 3 element Jump to : navigation , search For the other group formerly named Group III ( boron group ) , see Group 13 element .
Group 3 in the periodic table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur
Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton
Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium
Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium
Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury ( element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium
</item 10>
=== END ITEM id=7012a47a4d835481 ===

=== ITEM id=1579e890bccee064 ===
Gold sentences and facts:
Sentence 0: The story in the book takes place in Tulsa , Oklahoma , in 1965 , but this is never explicitly stated in the book .
  Fact 0.0: The story in the book takes place in Tulsa, Oklahoma.
  Fact 0.1: The story in the book takes place in 1965.
  Fact 0.2: The book never explicitly states that the story takes place in 1965.

Retrieved items:
<item 1>
The Outsiders (novel)
Paul Holden : A former friend of Darry 's . Other Characters Jerry Wood : The schoolteacher responsible for the children who were in the seemingly abandoned
church . Mr. Syme : Ponyboy 's English teacher who assigns him a theme to write that becomes the story of the novel itself . Controversy The Outsiders was a
controversial book at the time of its publication ; it is still currently challenged and debated . It was ranked # 38 on the American Library Association 's Top
100 Most Frequently Challenged Books of 1990 -- 1999 . This book has been banned from some schools and libraries because of the portrayal of gang violence ,
underage smoking and drinking , strong language / slang , and family dysfunction . However , in many U.S. schools , the book is part of the English curriculum
at the middle - or high - school level . See also The Outsiders House Museum References Jump up ^ Hinton , S.E. ( 2005 ) ( 1977 ) . `` speaking with S.E. Hinton
'' . The Outsiders . Speak / Penguin Putnam . p. 162 .
</item 1>

<item 2>
The Outsiders (novel)
The greasers run inside the burning church to save the children , but Ponyboy is rendered unconscious by the fumes . At the hospital he discovers that he and
Dally are not badly injured , but a piece of the church roof fell on Johnny and broke his back . Sodapop and Darry come to the hospital ; Darry breaks down and
cries . Ponyboy then realizes that Darry cares about him , and is only hard on Ponyboy because he loves him and cares about his future . The following morning
the newspapers declare Pony and Johnny heroes , but Johnny will be charged with manslaughter for Bob 's death . Two - Bit tells them that the greaser -- Soc
rivalry is to be settled in a final rumble . Ponyboy and Two - Bit are approached by a Soc named Randy , Bob 's best friend , who expresses remorse for his
involvement in the gang war , lacks confidence about the rumble ending the feud , and says he will not participate . Later , Ponyboy visits Johnny at the
hospital , where he is in critical condition . On their way home , Pony spots Cherry and they talk .
</item 2>

<item 3>
The Outsiders (novel)
Jump up ^ Cummings , Siena ( 2017 ) . `` ( www.standard.net/TX/2017/09/24/After-50-years-Outsiders After 50 years , ' Outsiders ' continues to ' stay gold ' ) .
'' Standard - Examiner . Ogden Publishing Corporation . Bibliography Hinton , S.E. ( 1997 ) . The Outsiders . New York : SPEAK . ISBN 9780140385724 . External
links S.E. Hinton 's Website `` ' The Outsiders ' : 40 Years Later '' By Dale Peck The New York Times September 23 , 2007 S.E. Hinton Books The Outsiders Rumble
Fish That Was Then , This Is Now Tex Taming the Star Runner Hawkes Harbor Some of Tim 's Stories Big David , Little David The Puppy Sister Books I Love Best
Yearly : Older Readers Award The Secret Diary of Adrian Mole , Aged 133⁄4 by Sue Townsend ( 1990 ) The Outsiders by S.E. Hinton ( 1991 ) Where 's Wally ?
</item 3>

<item 4>
The Outsiders (novel)
The next night , Ponyboy and two greaser friends , the hardened Dally and the quiet Johnny , meet Cherry and Marcia , a pair of Soc girls , at a drive - in
movie theater . Cherry spurns Dally 's rude advances , but Ponyboy ends up speaking civilly with Cherry , emotionally connecting with a Soc for the first time
in his life . Afterward , Ponyboy , Johnny , and their wisecracking friend Two - Bit begin to walk Cherry and Marcia home , when they are stopped by Cherry 's
boyfriend Bob , who badly beat up Johnny a few months back . Bob and the greasers exchange taunts , but Cherry prevents a fight by willingly leaving with Bob .
Ponyboy gets home at two in the morning , enraging Darry until he suddenly slaps Ponyboy . Ponyboy runs out the door and meets up with Johnny , expressing his
anger at Darry 's increasing coldness in the wake of his parents ' recent deaths in a car crash . Ponyboy and Johnny wander into a park , where Bob and four
other Socs surround them .
</item 4>

<item 5>
The Outsiders (novel)
After some heated talk , Ponyboy spits at the Socs , prompting them to try to drown him in a nearby fountain , but Johnny stabs Bob , killing him and dispersing
the Socs . Terrified of what to do next , Ponyboy and Johnny rush to find Dally , who gives them money and a loaded gun , directing them to hide in an abandoned
church in Windrixville . During their stay there , Pony cuts and dyes his hair as a disguise , reads Gone with the Wind to Johnny , and , upon viewing a
beautiful sunrise , recites the poem `` Nothing Gold Can Stay '' by Robert Frost . Days later , Dally comes to check on them , revealing that violence between
the greasers and Socs has escalated since Bob 's death into all - out city - wide warfare , with Cherry acting out of guilt as a spy for the greasers . Johnny
decides to turn himself in and Dally agrees to take the boys back home . As they attempt to leave , they notice the church has caught fire and several local
schoolchildren have become trapped inside .
</item 5>

<item 6>
The Outsiders (novel)
He is the unofficial leader of the gang . Johnny Cade : Ponyboy 's best friend , 16 years old , who is extremely quiet and lives with his alcoholic and abusive
mother and father . Dallas `` Dally '' Winston : The roughest of the greasers , who lived on the streets of New York City for three years . Keith `` Two - Bit
'' Mathews : A wise - cracking kleptomaniac . Steve Randle : Sodapop 's best friend since grade school . Timothy `` Tim '' Shepard : Leader of another greaser
gang : an organized downtown one . Sandy : Sodapop 's girlfriend . Socs Sherri `` Cherry '' Valance : Bob 's girlfriend , attends the same high school as
Ponyboy . Robert `` Bob '' Sheldon : Cherry 's boyfriend who is stabbed and killed by Johnny . Randy Adderson : A friend of Bob 's and Marcia 's boyfriend .
Marcia : Cherry 's best friend and Randy 's girlfriend . David : A member of Bob and Randy 's gang who attempted to drown Ponyboy in the fountain .
</item 6>

<item 7>
The Outsiders (novel)
On their way home , Pony spots Cherry and they talk . Cherry says she is unwilling to visit Johnny in the hospital because he killed her boyfriend . Pony calls
her a traitor , but after explaining herself , they end on good terms . After escaping the hospital , Dally shows up just in time for the rumble . The greasers
win the brutal fight . Afterward , Pony and Dally hurry back to the hospital to see Johnny , but he dies moments later and a maniacal Dally runs out of the room
. Pony returns home that night feeling confused and disoriented . Dally calls the house to say that he has robbed a store and is running from the police . The
greasers find Dally deliberately pointing an unloaded gun at the police , causing them to shoot and kill him . Overwhelmed , Ponyboy faints and is sick in bed
for several days due to the resulting concussion . When the hearing finally comes , the judge frees Ponyboy from responsibility for Bob 's death and allows Pony
to remain at home with Darry and Soda . Ponyboy returns to school , but his grades drop .
</item 7>

<item 8>
The Outsiders (novel)
The Outsiders ( novel ) - wikipedia The Outsiders ( novel ) Jump to : navigation , search The Outsiders First hardcover edition , 1967 Author S.E. Hinton Cover
artist Robert Hunt Country United States Language English Genre Young adult fiction Publisher Viking Press , Dell Publishing Publication date April 24 , 1967
Media type Print ( hardcover , paperback ) , Audiobook Pages 192 ISBN 0 - 670 - 53257 - 6 OCLC 64396432 Followed by That Was Then , This Is Now The Outsiders is
a coming - of - age novel by S.E. Hinton , first published in 1967 by Viking Press . Hinton was 15 when she started writing the novel , but did most of the work
when she was 16 and a junior in high school . Hinton was 18 when the book was published . The book details the conflict between two rival gangs divided by their
socioeconomic status : the working - class `` greasers '' and the upper - class `` Socs '' ( pronounced / ˈsoʊʃɪz / -- short for Socials ) .
</item 8>

<item 9>
The Outsiders (novel)
Ponyboy returns to school , but his grades drop . Although he is failing English , his teacher , Mr. Syme , says he will pass him if he writes a decent theme .
In the copy of Gone with the Wind that Johnny gave him before dying , Ponyboy finds a note from Johnny describing how he will die proudly after saving the kids
from the fire . Johnny also urges Ponyboy to `` stay gold '' . Ponyboy decides to write his English assignment about the recent events , and begins his essay
with the opening line of the novel : `` When I stepped out into the bright sunlight from the darkness of the movie house , I had only two things on my mind :
Paul Newman and a ride home ... '' Major Characters Greasers Ponyboy Curtis : Narrator and the youngest Curtis brother , 14 years old , who gets good grades and
runs track . Sodapop Curtis : The middle Curtis brother , 16 years old , a high school dropout who works at a gas station . Darrel `` Darry '' Curtis : The
oldest of the Curtis brothers , 20 years old , who has been caring for his brothers since their parents died in a car crash .
</item 9>

<item 10>
The Outsiders (novel)
The story is told in first - person perspective by teenaged protagonist Ponyboy Curtis . The story in the book takes place in Tulsa , Oklahoma , in 1965 , but
this is never explicitly stated in the book . A film adaptation was produced in 1983 , and a little - known short - lived television series appeared in 1990 ,
picking up where the movie left off . A stage adaptation was written by Christopher Sergel and published in 1990 . Contents ( hide ) 1 Plot summary 2 Major
characters 2.1 Greasers 2.2 Socs 2.3 Other characters 3 Controversy 4 See also 5 References 6 Bibliography 7 External links Plot summary Ponyboy Curtis , a
teenaged member of a loose gang of `` greasers '' , is leaving a movie theater when he is jumped by `` Socs '' , the greasers ' rival gang . Several greasers ,
including Ponyboy 's two older brothers -- the paternal Darry and the popular Sodapop -- come to his rescue .
</item 10>
=== END ITEM id=1579e890bccee064 ===

=== ITEM id=16cfbe4a3666f3bd ===
Gold sentences and facts:
Sentence 0: The culture of Australia is a Western culture , derived primarily from Britain but also influenced by the unique geography of Australia , the diverse input of Aboriginal , Torres Strait Islander and other Australian people . The British colonization of Australia began in 1788 , and waves of multi-ethnic migration followed . Evidence of a significant Anglo - Celtic heritage includes the predominance of the English language , the existence of a democratic system of government drawing upon the British traditions of Westminster Government , Parliamentarianism and constitutional monarchy , American constitutionalist and federalist traditions , Christianity as the dominant religion , and the popularity of sports originating in ( or influenced by ) the British Isles . Australian culture has diverged significantly since British settlement .
  Fact 0.0: The culture of Australia is a Western culture.
  Fact 0.1: The culture of Australia is derived primarily from Britain.
  Fact 0.2: The culture of Australia is also influenced by the unique geography of Australia.
  Fact 0.3: The culture of Australia is influenced by the diverse input of Aboriginal, Torres Strait Islander and other Australian people.
  Fact 0.4: British colonization of Australia began in 1788.
  Fact 0.5: Waves of multi-ethnic migration followed the British colonization of Australia.
  Fact 0.6: Evidence of a significant Anglo-Celtic heritage in Australia includes the predominance of the English language.
  Fact 0.7: Evidence of Anglo-Celtic heritage in Australia includes a democratic system of government drawing upon the British traditions of Westminster Government, Parliamentarianism and constitutional monarchy.
  Fact 0.8: Evidence of Anglo-Celtic heritage in Australia includes American constitutionalist and federalist traditions.
  Fact 0.9: Evidence of Anglo-Celtic heritage in Australia includes Christianity as the dominant religion.
  Fact 0.10: Evidence of Anglo-Celtic heritage in Australia includes the popularity of sports originating in (or influenced by) the British Isles.
  Fact 0.11: Australian culture has diverged significantly since British settlement.

Retrieved items:
<item 1>
Culture of Australia
The colonies established elected parliaments and rights for workers and women before most other Western nations . Federation in 1901 evidenced a growing sense
of national identity that had developed over the latter half of the 19th century , as seen in the works of the Heidelberg School painters and writers like Banjo
Paterson , Henry Lawson and Dorothea Mackellar . The World Wars profoundly altered Australia 's sense of identity , with World War I introducing the ANZAC
legend , and World War II seeing a reorientation from Britain to the United States as the nation 's foremost major ally . After the second war , 6.5 million
migrants from 200 nations brought immense new diversity , and Australians grew increasingly aware of their proximity to Asia . Over time , the diverse food ,
lifestyle and cultural practices of immigrants have been absorbed into mainstream Australian culture . Contents ( hide ) 1 Development of Australian culture 2
Symbols 3 Language 4 Humour 5 Arts 5.1 Literature 5.2 Theatre 5.3 Architecture 5.4 Visual arts 5.5 Cinema 5.6 Music 5.6. 1 Indigenous music 5.6.
</item 1>

<item 2>
Culture of Australia
Although Sydney was initially used by the British as a place of banishment for prisoners , the arrival of the British laid the foundations for Australia 's
democratic institutions and rule of law , and introduced the long traditions of English literature , Western art and music , and Judeo - Christian ethics and
religious outlook to a new continent . The British Empire expanded across the whole continent and established six colonies . The colonies were originally penal
colonies , with the exception of South Australia , which was established as a `` free colony '' with no convicts and a vision for a territory with political and
religious freedoms , together with opportunities for wealth through business and pastoral investments . Contact between the indigenous Australians and the new
settlers ranged from cordiality to violent conflict , but the diseases brought by Europeans were devastating to Aboriginal populations and culture . According
to the historian Geoffrey Blainey , during the colonial period : `` Smallpox , measles , influenza and other new diseases swept from one Aboriginal camp to
another ... The main conqueror of Aborigines was to be disease and its ally , demoralisation . '' William Wentworth established Australia 's first political
party in 1835 to demand democratic government for New South Wales .
</item 2>

<item 3>
Culture of Australia
By 2010 , the post-war immigration program had received more than 6.5 million migrants from every continent . The population tripled in the six decades to
around 21 million in 2010 , including people originating from 200 countries . More than 43 per cent of Australians were either born overseas or have one parent
who was born overseas . The population is highly urbanised , with more than 75 % of Australians living in urban centres , largely along the coast . Contemporary
Australia is a pluralistic society , rooted in liberal democratic traditions and espousing informality and egalitarianism as key societal values . While
strongly influenced by Anglo - Celtic origins , the culture of Australia has also been shaped by multi-ethnic migration which has influenced all aspects of
Australian life , including business , the arts , cuisine , sense of humour and sporting tastes . Contemporary Australia is also a culture that is profoundly
influenced by global movements of meaning and communication , including advertising culture . In turn , globalizing corporations from Holden to Exxon have
attempted to associate their brand with Australian cultural identity . This process intensified from the 1970s onwards .
</item 3>

<item 4>
Culture of Australia
The British colonization of Australia began in 1788 , and waves of multi-ethnic migration followed . Evidence of a significant Anglo - Celtic heritage includes
the predominance of the English language , the existence of a democratic system of government drawing upon the British traditions of Westminster Government ,
Parliamentarianism and constitutional monarchy , American constitutionalist and federalist traditions , Christianity as the dominant religion , and the
popularity of sports originating in ( or influenced by ) the British Isles . Australian culture has diverged significantly since British settlement . Aboriginal
people are believed to have arrived as early as 60,000 years ago , and evidence of Aboriginal art in Australia dates back at least 30,000 years . Several states
and territories had their origins as penal colonies , with the first British convicts arriving at Sydney Cove in 1788 . Stories of outlaws like the bushranger
Ned Kelly have endured in Australian music , cinema and literature . The Australian gold rushes from the 1850s brought wealth as well as new social tensions to
Australia , including the miners ' Eureka Stockade rebellion . The colonies established elected parliaments and rights for workers and women before most other
Western nations .
</item 4>

<item 5>
Culture of Australia
These wars profoundly affected Australia 's sense of nationhood and a proud military legend developed around the spirit of Australia 's ANZAC troops , who came
to symbolise the virtues of mateship , courage and endurance for the nation . The Australian colonies had a period of extensive multi-ethnic immigration during
the Australian gold rushes of the latter half of the 19th century , but following Federation in 1901 , the Parliament instigated the White Australia Policy that
gave preference to British migrants and ensured that Australia remained a predominantly Anglo - Celtic society until well into the 20th Century . The post-World
War II immigration program saw the policy dismantled by successive governments , permitting large numbers of Southern European , and later Asian and Middle
Eastern migrants to arrive . The Menzies Government ( 1949 - 1966 ) and Holt Government dismantled the legal barriers to multi-ethnic immigration and by the
1970s , the Whitlam and Fraser Governments were promoting multiculturalism . Countries of birth of Australian estimated resident population , 2006 Some States
and Territories of Australia retained discriminatory laws relating to voting rights for Aborigines into the 1960s , at which point full legal equality was
established .
</item 5>

<item 6>
Culture of Australia
Culture of Australia - Wikipedia Culture of Australia Jump to : navigation , search Part of a series on the Culture of Australia History People ( show )
Australian people African Australian Albanian Australian American Australian Anglo - Celtic Australian Asian Australian Assyrian Australian Bosnian Australian
Chinese Australian Cornish Australian English Australian European Australian Filipino Australian Greek Australian Indian Australian Indigenous Australian Irish
Australian Italian Australian Japanese Australian Jewish Australian Korean Australian Latin American Australian Lebanese Australian Macedonian Australian
Pakistani Australian Scottish Australian Serbian Australian South African Australian Sudanese Australian Taiwanese Australian Welsh Australian Languages
Traditions ( show ) Humour Mythology and folklore ( show ) Mythology folklore Cuisine Festivals Religion ( show ) Aboriginal Christianity ( Roman Catholic
Anglican Uniting Church Presbyterian Baptist Lutheran Coptic Orthodox more ) Buddhism Islam Sikhism Judaism Hinduism Bahá'í Faith Irreligion Art ( show )
Architecture Literature ( show ) Comic books Music and performing arts ( show ) Music Performing arts Media ( show ) Radio Television Cinema Sport Monuments (
show ) World Heritage Sites Symbols ( show ) Flag Coat of arms Australia portal The culture of Australia is a Western culture , derived primarily from Britain
but also influenced by the unique geography of Australia , the diverse input of Aboriginal , Torres Strait Islander and other Australian people .
</item 6>

<item 7>
Culture of Australia
1 Indigenous music 5.6. 2 Folk music and national songs 5.6. 3 Classical music 5.6. 4 Popular music 5.7 Dance 5.8 Television 6 Religion 7 Public holidays 8
Cuisine 8.1 Beverages 9 Clothing and apparel 10 Sport 10.1 Cricket 10.2 Football codes 10.3 Water sports 10.4 Other sports 11 Folklore 12 Attitudes , beliefs
and stereotypes 13 References Development of Australian culture ( edit ) Main article : History of Australia A Luritja man demonstrating method of attack with
boomerang under cover of shield ( 1920 ) The oldest surviving cultural traditions in Australia -- and some of the oldest surviving cultural traditions on earth
-- are those of Australia 's Aboriginal and Torres Strait Islander peoples . Their ancestors have inhabited Australia for between 40,000 and 60,000 years ,
living a hunter - gatherer lifestyle . In 2006 , the Indigenous population was estimated at 517,000 people , or 2.5 per cent of the total population .
</item 7>

<item 8>
Culture of Australia
Multicultural immigration has increased Australia 's religious diversity . The proportion of the total population who are Christian fell from 71 % in 1996 to
around 61.1 % in 2011 , while people affiliated with non-Christian religions increased from around 3.5 % to 7.2 % over the same period . Buddhism increased most
rapidly from 1.1 % to 2.5 % . Increased immigration from South - East Asia has been a major factor in this growth , but Australians of Anglo - Celtic origin
have also shown increasing interest in Buddhism . Islam increased during the period from 1.1 % to 2.2 % with diverse communities concentrated mainly in Sydney
and Melbourne . The history of the Jews in Australia dates back to the First Fleet , which brought Jewish convicts to Sydney in 1788 . Today , many Jews in
Australia originated as refugees and Holocaust survivors who arrived during and after World War II . Hindus came to Australia as labourers and merchants during
the 19th century and numbers increased dramatically from the 1960s , more than doubling between 1996 and 2006 .
</item 8>

<item 9>
Culture of Australia
Most Aboriginal Australians have a belief system based on the Dreaming , or Dreamtime , which refers both to a time when ancestral spirits created land and
culture , and to the knowledge and practices that define individual and community responsibilities and identity . Conflict and reconciliation between indigenous
and non-indigenous Australians has been a source of much art and literature in Australia , and ancient Aboriginal artistic styles and iconic inventions such as
the boomerang , the didgeridoo and Indigenous Australian music have become symbols of modern Australia . Governor Arthur Phillip hoists the British flag over
the new colony at Sydney in 1788 South Australian suffragette Catherine Helen Spence ( 1825 -- 1910 ) . The Australian colonies established democratic
parliaments from the 1850s and began to grant women the vote in the 1890s . The arrival of the first British settlers at what is now Sydney in 1788 introduced
Western civilisation to the Australian continent .
</item 9>

<item 10>
Culture of Australia
A 1967 referendum to include all Aborigines in the national electoral roll census was overwhelmingly approved by voters . In 1984 , a group of Pintupi people
who were living a traditional hunter - gatherer desert - dwelling life were tracked down in the Gibson Desert and brought into a settlement . They are believed
to have been the last uncontacted tribe . While the British cultural influence remained strong into the 21st century , other influences became increasingly
important . The Hawaiian sport of surfing was adopted in Australia where a beach culture and the locally developed surf lifesaving movement was already
burgeoning in the early 20th century . American pop culture and cinema were embraced in the 20th century , with country music and later rock and roll sweeping
Australia , aided by the new technology of television and a host of American content . The 1956 Melbourne Olympics announced a confident , prosperous post-war
nation , and new cultural icons like Australian country music star Slim Dusty and dadaist Barry Humphries expressed a uniquely Australian identity . Australia
's contemporary immigration program has two components : a program for skilled and family migrants and a humanitarian program for refugees and asylum seekers .
</item 10>
=== END ITEM id=16cfbe4a3666f3bd ===

=== ITEM id=6284a795417c85bd ===
Gold sentences and facts:
Sentence 0: Dollree Mapp was an employee in the illegal gambling rackets dominated by Cleveland rackets kingpin Shon Birns . On May 23 , 1957 , police officers in Cleveland , Ohio , received an anonymous tip by phone that Virgil Ogletree , a numbers operator who was wanted for questioning in the bombing of rival numbers racketeer and future boxing promoter Don King 's home three days earlier , might be found at Mapp 's house , as well as illegal betting slips and equipment employed in the `` California Gold '' numbers operation set up by Mapp 's boyfriend Edward Keeling . Three officers went to the home and asked for permission to enter , but Mapp , after consulting her lawyer by telephone , refused to admit them without a search warrant . Two officers left , and one remained , watching the house from across the street .
  Fact 0.0: Dollree Mapp was an employee in the illegal gambling rackets.
  Fact 0.1: The illegal gambling rackets were dominated by Cleveland rackets kingpin Shon Birns.
  Fact 0.2: On May 23, 1957, police officers in Cleveland, Ohio, received an anonymous tip by phone.
  Fact 0.3: The anonymous tip said Virgil Ogletree might be found at Dollree Mapp's house.
  Fact 0.4: Virgil Ogletree was a numbers operator.
  Fact 0.5: Virgil Ogletree was wanted for questioning in the bombing of Don King's home three days before the tip.
  Fact 0.6: Don King was a rival numbers racketeer.
  Fact 0.7: Don King was a future boxing promoter.
  Fact 0.8: The anonymous tip said illegal betting slips might be found at Dollree Mapp's house.
  Fact 0.9: The anonymous tip said equipment employed in the "California Gold" numbers operation might be found at Dollree Mapp's house.
  Fact 0.10: The "California Gold" numbers operation was set up by Dollree Mapp's boyfriend Edward Keeling.
  Fact 0.11: Edward Keeling was Dollree Mapp's boyfriend.
  Fact 0.12: Three officers went to Dollree Mapp's home.
  Fact 0.13: The three officers asked for permission to enter the home.
  Fact 0.14: Dollree Mapp consulted her lawyer by telephone.
  Fact 0.15: Dollree Mapp refused to admit the officers without a search warrant.
  Fact 0.16: Two officers left.
  Fact 0.17: One officer remained, watching the house from across the street.

Retrieved items:
<item 1>
Mapp v. Ohio
23 , 1957 , p. 1 . Jump up ^ http://www.infoplease.com/us/supreme-court/cases/ar19.html Further reading ( edit ) Long , Carolyn ( 2006 ) . Mapp v. Ohio :
Guarding Against Unreasonable Searches and Seizures . University Press of Kansas . ISBN 0 - 7006 - 1441 - 9 . Stewart , Potter ( 1983 ) . `` The Road to Mapp v.
Ohio and beyond : The Origins , Development and Future of the Exclusionary Rule in Search - and - Seizure Cases '' . Columbia Law Review . 83 ( 6 ) : 1365 --
1404 . doi : 10.2307 / 1122492 . JSTOR 1122492 . Zotti , Priscilla H. Machado ( 2005 ) . Injustice for All : Mapp vs. Ohio and the Fourth Amendment . New York :
Peter Lang . ISBN 0 - 8204 - 7267 - 0 .
</item 1>

<item 2>
Mapp v. Ohio
Supreme Court voted 5 - 3 in favor of Mapp . The Court overturned the conviction , and five justices found that the States were bound to exclude evidence seized
in violation of the 4th Amendment . This ruling officially applied the exclusionary rule to the states . See also ( edit ) United States Bill of Rights Fruit of
the poisonous tree Exclusionary rule References ( edit ) Jump up ^ Zotti , Priscilla . Injustice for All ( Peter Lang , 2005 ) . Jump up ^ Mapp v. Ohio , 367
U.S. 643 ( U.S. 1961 ) . Jump up ^ Duignan , Brian ( 2012 - 05 - 25 ) . `` Mapp v. Ohio '' . Encyclopædia Britannica Online . Encyclopædia Britannica Inc .
Retrieved March 19 , 2014 . Jump up ^ https://www.law.cornell.edu/supremecourt/text/367/643#writing - Jump up ^ Williams , Bob . `` Dolly Mapp Sheds Glamour Gal
Role '' , Cleveland Call and Post , Nov.
</item 2>

<item 3>
Mapp v. Ohio
§ 1983 Monroe v. Pape ( 1961 ) Malley v. Briggs ( 1986 ) ( hide ) Incorporation against States Unreasonable search and seizure Wolf v. Colorado ( 1949 ) Rochin
v. California ( 1952 ) Irvine v. California ( 1954 ) Elkins v. United States ( 1960 ) Mapp v. Ohio ( 1961 ) Warrant requirements Ker v. California ( 1963 )
Aguilar v. Texas ( 1964 ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=Mapp_v._Ohio&oldid=819300745 '' Categories : United States Supreme Court
decisions that overrule a prior Supreme Court decision Cuyahoga County , Ohio United States Supreme Court cases United States Fourth Amendment case law 20th
century American trials 1961 in United States case law Incorporation case law American Civil Liberties Union litigation United States Supreme Court cases of the
Warren Court Cleveland Division of Police Hidden categories : Articles needing additional references from October 2017 All articles needing additional
references Talk Contents About Wikipedia Simple English Edit links This page was last edited on 8 January 2018 , at 16 : 31 .
</item 3>

<item 4>
Mapp v. Ohio
Mapp v. Ohio - wikipedia Mapp v. Ohio Jump to : navigation , search Mapp v. Ohio Supreme Court of the United States Argued March 29 , 1961 Decided June 19 ,
1961 Full case name Dollree Mapp v. State of Ohio Citations 367 U.S. 643 ( more ) 81 S. Ct. 1684 ; 6 L. Ed . 2d 1081 ; 1961 U.S. LEXIS 812 ; 86 Ohio L. Abs. 513
; 16 Ohio Op . 2d 384 ; 84 A.L.R. 2d 933 Prior history Defendant convicted , Cuyahoga County , Ohio Court of Common Pleas ; affirmed , Ohio Court of Appeals ;
affirmed , Ohio Supreme Court 166 N.E. 2d 387 ( Ohio 1960 ) Subsequent history Rehearing denied , 368 U.S. 871 ( 1961 ) Holding The Fourth Amendment prohibition
against unreasonable searches and seizures , as applied to the states through the Fourteenth , excludes unconstitutionally obtained evidence from use in
criminal prosecutions . Ohio Supreme Court reversed .
</item 4>

<item 5>
Mapp v. Ohio
On May 23 , 1957 , police officers in Cleveland , Ohio , received an anonymous tip by phone that Virgil Ogletree , a numbers operator who was wanted for
questioning in the bombing of rival numbers racketeer and future boxing promoter Don King 's home three days earlier , might be found at Mapp 's house , as well
as illegal betting slips and equipment employed in the `` California Gold '' numbers operation set up by Mapp 's boyfriend Edward Keeling . Three officers went
to the home and asked for permission to enter , but Mapp , after consulting her lawyer by telephone , refused to admit them without a search warrant . Two
officers left , and one remained , watching the house from across the street . Thirteen hours later , four cars full of police arrived and knocked on the door .
When she did n't answer , they forced the door . Mapp asked to see the alleged warrant and was shown a piece of paper which she snatched away from an officer ,
putting it inside her dress . The officers struggled with Mapp and recovered the piece of paper which was not seen by her or her lawyers again , and was not
introduced as evidence in any of the ensuing court proceedings .
</item 5>

<item 6>
Mapp v. Ohio
When asked about the warrant during oral argument at the Supreme Court , the Cleveland prosecutor arguing the case cautiously deflected the question , which the
court did not press . As the search of Mapp 's second - floor , 2 - bedroom apartment began , police handcuffed her for being belligerent . The police searched
the house thoroughly , and discovered Ogletree , who was subsequently cleared on the bombing charge , hiding in the apartment of the downstairs tenant , Minerva
Tate . In the search of Mapp 's apartment and in a footlocker in the basement of the house police found a quantity of `` California Gold '' betting slips and
paraphernalia . They also found a pistol and a small quantity of pornographic books and pictures which Mapp stated a previous tenant named Morris Jones had left
behind .
</item 6>

<item 7>
Mapp v. Ohio
The Supreme Court accomplished this by use of a principle known as selective incorporation ; in Mapp this involved the incorporation of the provisions , as
interpreted by the Court , of the Fourth Amendment which are applicable only to actions of the federal government into the Fourteenth Amendment due process
clause which is applicable to actions of the states . Contents ( hide ) 1 Circumstances of the case 2 Decision 3 See also 4 References 5 Further reading 6
External links Circumstances of the case ( edit ) This section needs additional citations for verification . Please help improve this article by adding
citations to reliable sources . Unsourced material may be challenged and removed . ( October 2017 ) ( Learn how and when to remove this template message )
Dollree Mapp was an employee in the illegal gambling rackets dominated by Cleveland rackets kingpin Shon Birns .
</item 7>

<item 8>
Mapp v. Ohio
Mapp was arrested , charged , and cleared on a misdemeanor charge of possessing numbers paraphernalia ; but several months later , after she refused to testify
against Shon Birns , Edward Keeling and their associates at their trial that October for the attempted shakedown of Don King , she was prosecuted for possession
of the books , found guilty at a 1958 trial of `` knowingly having had in her possession and under her control certain lewd and lascivious books , pictures ,
and photographs in violation of 2905.34 of Ohio 's Revised Code '' , and sentenced to one to seven years in prison . Mapp then appealed her case to the Supreme
Court , on the grounds that the police had no probable cause to suspect her of having the books . She stated that the 4th Amendment should be incorporated to
the state and local level . She argued that the police could n't use the books as evidence in trial because they were found without a warrant and therefore
illegally recovered . Mapp is the first to call upon this law . Decision ( edit ) The U.S. Supreme Court voted 5 - 3 in favor of Mapp .
</item 8>

<item 9>
Mapp v. Ohio
ISBN 0 - 8204 - 7267 - 0 . External links ( edit ) Wikisource has original text related to this article : Mapp v. Ohio Archival source documents relating to the
Mapp case at Cleveland Memory Dollree Mapp , Who Defied Police Search in Landmark Case , Is Dead - New York Times `` Supreme Court Landmark Case Mapp v. Ohio ''
from C - SPAN 's Landmark Cases : 12 Historic Supreme Court Decisions ( hide ) United States 4th Amendment case law Scope of the Fourth Amendment What
constitutes a search ? Ex parte Jackson ( 1878 ) Boyd v. United States ( 1886 ) Gouled v. United States ( 1921 ) Burdeau v. McDowell ( 1921 ) Hester v. United
States ( 1924 ) United States v. Lee ( 1927 ) Olmstead v. United States ( 1928 ) Goldman v. United States ( 1942 ) On Lee v. United States ( 1952 ) Abel v.
United States ( 1960 ) Silverman v. United States ( 1961 ) Lewis v.
</item 9>

<item 10>
Mapp v. Ohio
Ohio Supreme Court reversed . Court membership Chief Justice Earl Warren Associate Justices Hugo Black Felix Frankfurter William O. Douglas Tom C. Clark John M.
Harlan II William J. Brennan , Jr . Charles E. Whittaker Potter Stewart Case opinions Majority Clark , joined by Warren , Black , Douglas , Brennan Concurrence
Black Concurrence Douglas Concurrence Stewart Dissent Harlan , joined by Frankfurter , Whittaker Laws applied U.S. Const . amends. IV , XIV This case overturned
a previous ruling or rulings Wolf v. Colorado ( 1949 ) Mapp v. Ohio , 367 U.S. 643 ( 1961 ) , was a landmark case in criminal procedure , in which the United
States Supreme Court decided that evidence obtained in violation of the Fourth Amendment , which protects against `` unreasonable searches and seizures , '' may
not be used in state law criminal prosecutions in state courts , as well as in federal criminal law prosecutions in federal courts as had previously been the
law .
</item 10>
=== END ITEM id=6284a795417c85bd ===

=== ITEM id=d8877ac9ef89a0fc ===
Gold sentences and facts:
Sentence 0: In graph theory , a loop ( also called a self - loop or a `` buckle '' ) is an edge that connects a vertex to itself . A simple graph contains no loops .
  Fact 0.0: In graph theory, a loop is an edge that connects a vertex to itself
  Fact 0.1: A loop is also called a self-loop
  Fact 0.2: A loop is also called a buckle
  Fact 0.3: A simple graph contains no loops

Retrieved items:
<item 1>
Vanishing point
A , B , C , D and V are points on the image , their separation given in pixels ; A ' , B ' , C ' and D ' are in the real world , their separation in metres . In
( 1 ) , the width of the side street , W is computed from the known widths of the adjacent shops . In ( 2 ) , the width of only one shop is needed because a
vanishing point , V is visible . Camera calibration : The vanishing points of an image contain important information for camera calibration . Various
calibration techniques have been introduced using the properties of vanishing points to find intrinsic and extrinsic calibration parameters . 3D reconstruction
: A man - made environment has two main characteristics -- several lines in the scene are parallel , and a number of edges present are orthogonal . Vanishing
points aid in comprehending the environment . Using sets of parallel lines in the plane , the orientation of the plane can be calculated using vanishing points
. Torre and Coelho performed extensive investigation in the use of vanishing points to implement a full system .
</item 1>

<item 2>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 2>

<item 3>
Vanishing point
The horizon line is a theoretical line that represents the eye level of the observer . If the object is below the horizon line , its vanishing lines angle up to
the horizon line . If the object is above , they slope down . All vanishing lines end at the horizon line . Properties of vanishing points ( edit ) 1 .
Projections of two sets of parallel lines lying in some plane π appear to converge , i.e. the vanishing point associated with that pair , on a horizon line , or
vanishing line H formed by the intersection of the image plane with the plane parallel to π and passing through the pinhole . Proof : Consider the ground plane
π , as y = c which is , for the sake of simplicity , orthogonal to the image plane . Also , consider a line L that lies in the plane π , which is defined by the
equation ax + bz = d .
</item 3>

<item 4>
Vanishing point
5 D Bird 's - eye view Cross section Cutaway drawing Exploded view drawing Fisheye lens Panorama Worm 's - eye view Zoom lens Topics ( show ) 3D projection
Anamorphosis Axonometry Computer graphics Computer - aided design Descriptive geometry Engineering drawing Map projection Picture plane Plans ( drawings )
Projection ( linear algebra ) Projection plane Projective geometry Stereoscopy Technical drawing Vanishing point Video game graphics Viewing frustum Contents (
hide ) 1 Vector notation 2 Theorem 3 Vanishing line 4 Properties of vanishing points 5 Curvilinear and reverse perspective 6 Detection of vanishing points 7
Applications of vanishing points 8 See also 9 References 10 External links Vector notation ( edit ) A 2D construction of perspective viewing , showing the
formation of a vanishing point The vanishing point may also be referred to as the `` direction point '' , as lines having the same directional vector , say D ,
will have the same vanishing point or converge at the same vanishing points .
</item 4>

<item 5>
Loop (graph theory)
Such a graph is called a bouquet . Contents ( hide ) 1 Degree 2 See also 3 References 4 External links Degree ( edit ) For an undirected graph , the degree of a
vertex is equal to the number of adjacent vertices . A special case is a loop , which adds two to the degree . This can be understood by letting each connection
of the loop edge count as its own adjacent vertex . In other words , a vertex with a loop `` sees '' itself as an adjacent vertex from both ends of the edge
thus adding two , not one , to the degree . For a directed graph , a loop adds one to the in degree and one to the out degree . See also ( edit ) Loops in Graph
Theory Cycle ( graph theory ) Graph theory Glossary of graph theory Loops in Topology Möbius ladder Möbius strip Strange loop Klein bottle References ( edit )
Balakrishnan , V.K. ; Graph Theory , McGraw - Hill ; 1 edition ( February 1 , 1997 ) . ISBN 0 - 07 - 005489 - 4 .
</item 5>

<item 6>
Vanishing point
Vanishing point - wikipedia Vanishing point Jump to : navigation , search For other uses , see Vanishing point ( disambiguation ) . A vanishing point can be
seen at the far end of this railroad . A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective
projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a
picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the
image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three
vanishing points . Part of a series on Graphical projection Planar ( show ) Parallel projection Orthographic projection Multiview projection Axonometric
projection Isometric projection Oblique projection Perspective projection Curvilinear perspective Reverse perspective Views ( show ) 2.
</item 6>

<item 7>
Vanishing point
She notes , in terms of projective geometry , the vanishing point is the image of the point at infinity associated with L , as the sightline from O through the
vanishing point is parallel to L . Vanishing line ( edit ) As a vanishing point originates in a line , so a vanishing line originates in a plane α that is not
parallel to the picture π . Given the eye point O , and β the plane parallel to α and lying on O , then the vanishing line of α is β ∩ π . For example , when α
is the ground plane and β is the horizon plane , then the vanishing line of α is the horizon line β ∩ π . Anderson notes , `` Only one particular vanishing line
occurs , often referred to as the `` horizon '' . To put it simply , the vanishing line of some plane , say α , is obtained by the intersection of the image
plane with another plane , say β , parallel to the plane of interest ( α ) , passing through the camera center . For different sets of lines parallel to this
plane α , their respective vanishing points will lie on this vanishing line .
</item 7>

<item 8>
Loop (graph theory)
Loop ( graph Theory ) - wikipedia Loop ( graph Theory ) Jump to : navigation , search A graph with a loop on vertex 1 In graph theory , a loop ( also called a
self - loop or a `` buckle '' ) is an edge that connects a vertex to itself . A simple graph contains no loops . Depending on the context , a graph or a
multigraph may be defined so as to either allow or disallow the presence of loops ( often in concert with allowing or disallowing multiple edges between the
same vertices ) : Where graphs are defined so as to allow loops and multiple edges , a graph without loops or multiple edges is often distinguished from other
graphs by calling it a `` simple graph '' . Where graphs are defined so as to disallow loops and multiple edges , a graph that does have loops or multiple edges
is often distinguished from the graphs that satisfy these constraints by calling it a `` multigraph '' or `` pseudograph '' . In a graph with one vertex , all
edges must be loops . Such a graph is called a bouquet .
</item 8>

<item 9>
Vanishing point
If we know the coordinates of one of these points , say v , and the direction of a straight line on the image plane , which passes through a second point , say
v , we can compute the coordinates of both v and v 3 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x ,
y , f ) , v ≡ ( x , y , f ) be the three corresponding vanishing points respectively . The orthocenter of the triangle with vertices in the three vanishing
points is the intersection of the optical axis and the image plane . Curvilinear and reverse perspective ( edit ) A curvilinear perspective is a drawing with
either 4 or 5 vanishing points . In 5 - point perspective the vanishing points are mapped into a circle with 4 vanishing points at the cardinal headings N , W ,
S , E and one at the circle 's origin . A reverse perspective is a drawing with vanishing points that are placed outside the painting with the illusion that
they are `` in front of '' the painting . Single point perspective projection . Double point perspective projection .
</item 9>

<item 10>
Typeface anatomy
ij each have a dot , jot , or tittle . Angles of strokes are called apices if at the top and vertices if at the bottom . w has one apex and two vertices ; v has
one vertex . The font shown in the example is stressed : strokes have varying widths . In this example , the stroke at the top of the g is thinner at the top
and bottom than on the sides -- a vertical stress . Terminals ( edit ) The terminals ( ends ) of instrokes and outstrokes often end in serif or sans - serif . A
seriffed terminal may be described as a wedge , bulbous , teardrop , slab , etc. , depending on the design of the type . Sorting typefaces can be practical
based on its look . A clue feature is to define the serif style . Some designs also have spurs , which are smaller than serifs and appear on angles rather than
at a terminal , as on e or G . Space ( edit ) Areas of negative space ( white space ) formed by straight or curved strokes are called counters . Closed counters
are found in abdegopq ABDOPQR , and open counters in acefhmnrstu .
</item 10>
=== END ITEM id=d8877ac9ef89a0fc ===
