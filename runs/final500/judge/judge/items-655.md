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

=== ITEM id=aab1fd2536cd111f ===
Gold sentences and facts:
Sentence 0: From September 1947 , Toyota 's small - sized vehicles were sold under the name `` Toyopet '' ( トヨペット ) . The first vehicle sold under this name was the Toyopet SA , but it also included vehicles such as the Toyopet SB light truck , Toyopet Stout light truck , Toyopet Crown , Toyopet Master , and the Toyopet Corona . The word `` Toyopet ( Japanese article ) '' was a nickname given to the Toyota SA due to its small size , as the result of a naming contest the Toyota Company organized in 1947 . However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The name was soon dropped for the American market , but continued in other markets until the mid-1960s .
  Fact 0.0: From September 1947, Toyota's small-sized vehicles were sold under the name 'Toyopet' (トヨペット).
  Fact 0.1: The first vehicle sold under the name Toyopet was the Toyopet SA.
  Fact 0.2: Toyopet vehicles also included the Toyopet SB light truck.
  Fact 0.3: Toyopet vehicles also included the Toyopet Stout light truck.
  Fact 0.4: Toyopet vehicles also included the Toyopet Crown.
  Fact 0.5: Toyopet vehicles also included the Toyopet Master.
  Fact 0.6: Toyopet vehicles also included the Toyopet Corona.
  Fact 0.7: The word 'Toyopet' was a nickname given to the Toyota SA due to its small size.
  Fact 0.8: The Toyopet nickname resulted from a naming contest the Toyota Company organized in 1947.
  Fact 0.9: When Toyota entered the American market in 1957 with the Crown, the Toyopet name was not well received due to connotations of toys and pets.
  Fact 0.10: The Toyopet name was soon dropped for the American market.
  Fact 0.11: The Toyopet name continued in other markets until the mid-1960s.

Retrieved items:
<item 1>
Toyota
In the first three months of 2007 , Toyota together with its half - owned subsidiary Daihatsu reported number one sales of 2.348 million units . Toyota 's brand
sales had risen 9.2 % largely on demand for Corolla and Camry sedans . The difference in performance was largely attributed to surging demand for fuel -
efficient vehicles . In November 2006 , Toyota Motor Manufacturing Texas added a facility in San Antonio . Toyota has experienced quality problems and was
reprimanded by the government in Japan for its recall practices . In 2007 , Toyota maintained over 16 % of the US market share and was listed second only to GM
in terms of volume . Toyota Century Royal is the official state car of the Japanese imperial family , namely for the current Emperor of Japan . Toyota was hit
by the global financial crisis of 2008 as it was forced in December 2008 to forecast its first annual loss in 70 years . In January 2009 , it announced the
closure of all of its Japanese plants for 11 days to reduce output and stocks of unsold vehicles .
</item 1>

<item 2>
Toyota
Toyota Technical Center , Ann Arbor Charter Township , Michigan Toyota has a large presence in the United States with six major assembly plants in Huntsville ,
Alabama , Georgetown , Kentucky , Princeton , Indiana , San Antonio , Texas , Buffalo , West Virginia , and Blue Springs , Mississippi . Toyota had a joint -
venture operation with General Motors at New United Motor Manufacturing Inc. in Fremont , California , which began in 1984 and ended in 2009 . It still has a
joint venture with Subaru at Subaru of Indiana Automotive , Inc. in Lafayette , Indiana , which started in 2006 . In these assembly plants , the Camry and the
Tundra are manufactured , among others . Toyota marketing , sales , and distribution in the US are conducted through a separate subsidiary , Toyota Motor Sales
, U.S.A. , Inc . It has started producing larger trucks , such as the new Tundra , to go after the large truck market in the United States . Toyota is also
pushing hybrid vehicles in the US such as the Prius , Camry Hybrid , Highlander Hybrid , and various Lexus products . Currently , Toyota has no plans to offer
diesel motor options in its North American products , including the light - duty pickup trucks .
</item 2>

<item 3>
Toyota
Akio Toyoda became the new president and CEO of the company on June 23 , 2009 , by replacing Katsuaki Watanabe , who became the new vice chairman by replacing
Katsuhiro Nakagawa . Worldwide presence ( edit ) The Toyota Camry is assembled in several facilities around the world including Australia , China , Taiwan ,
Japan , Malaysia , the Philippines , Pakistan , Russia , Thailand , India , Vietnam , and the United States Toyota has factories in most parts of the world ,
manufacturing or assembling vehicles for local markets in Japan , Australia , India , Sri Lanka , Canada , Indonesia , Poland , South Africa , Turkey , Colombia
, the United Kingdom , the United States , France , Brazil , Portugal , and more recently , Argentina , Czech Republic , Mexico , Malaysia , Thailand , Pakistan
, Egypt , China , Vietnam , Venezuela , the Philippines , and Russia .
</item 3>

<item 4>
Toyota
1990s ( edit ) In the 1990s , Toyota began to branch out from producing mostly compact cars by adding many larger and more luxurious vehicles to its lineup ,
including a full - sized pickup , the T100 ( and later the Tundra ) , several lines of SUVs , a sport version of the Camry , known as the Camry Solara , and the
Scion brand . They would also launch newer and arguably more iconic iterations of their sports cars , namely the MR2 , Celica , and Supra during this era , all
of which have already become icons of the 1990s . With a major presence in Europe , due to the success of Toyota Team Europe , the corporation decided to set up
Toyota Motor Europe Marketing and Engineering , TMME , to help market vehicles in the continent . Two years later , Toyota set up a base in the United Kingdom ,
TMUK , as the company 's cars had become very popular among British drivers . Bases in Indiana , Virginia , and Tianjin were also set up . In 1999 , the company
decided to list itself on the New York and London Stock Exchanges .
</item 4>

<item 5>
Toyota
However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The
name was soon dropped for the American market , but continued in other markets until the mid-1960s . 1960s -- 1970s ( edit ) By the early 1960s , the US had
begun placing stiff import tariffs on certain vehicles . The so - called `` chicken tax '' of 1964 placed a 25 % tax on imported light trucks . In response to
the tariff , Toyota , Nissan Motor Co. and Honda Motor Co. began building plants in the US by the early 1980s . 1980s ( edit ) With over 40 million sold , the
Corolla is one of the most popular and best selling cars in the world . Toyota received its first Japanese Quality Control Award at the start of the 1980s and
began participating in a wide variety of motorsports . Due to the 1973 oil crisis , consumers in the lucrative US market began turning to small cars with better
fuel economy .
</item 5>

<item 6>
Toyota
The principle of jidoka , which means the machine stops itself when a problem occurs , became later a part of the Toyota Production System . Looms were built on
a small production line . In 1929 , the patent for the automatic loom was sold to the British company Platt Brothers , generating the starting capital for the
automobile development . Mass production of Toyoda automated loom , displayed at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi Pref . Japan The
production of Toyota automobiles was started in 1933 as a division of Toyoda Automatic Loom Works devoted to the production of automobiles under the direction
of the founder 's son , Kiichiro Toyoda . Its first vehicles were the A1 passenger car and the G1 in 1935 . The Toyota Motor Co. was established as an
independent company in 1937 Toyoda Standard Sedan AA 1936 Vehicles were originally sold under the name `` Toyoda '' ( トヨダ ) , from the family name of the
company 's founder , Kiichirō Toyoda . In April 1936 , Toyoda 's first passenger car , the Model AA , was completed .
</item 6>

<item 7>
Toyota
American car manufacturers had considered small economy cars to be an entry - level product , and their small vehicles employed a low level of quality to keep
the price low . Conservative Toyota held on to rear - wheel - drive designs for longer than most ; while a clear first in overall production they were only
third in production of front - wheel - drive cars in 1983 , behind Nissan and Honda . In part due to this , Nissan 's Sunny managed to squeeze by the Corolla in
numbers built that year . In 1982 , the Toyota Motor Company and Toyota Motor Sales merged into one company , the Toyota Motor Corporation . Two years later ,
Toyota entered into a joint venture with General Motors called the New United Motor Manufacturing , Inc , NUMMI , operating an automobile - manufacturing plant
in Fremont , California . The factory was an old General Motors plant that had been closed for two years . Toyota then started to establish new brands at the
end of the 1980s , with the launch of their luxury division Lexus in 1989 .
</item 7>

<item 8>
Toyota
In Chinese - speaking countries using simplified Chinese characters ( e.g. China ) , Toyota is known as `` 丰田 '' ( pronounced as Fēngtián in Mandarin Chinese
and Hɔng Tshan in Minnanese ) . These are the same characters as the founding family 's name `` Toyoda '' in Japanese . Logo until 1989 , now still used as a
corporate logo Logo since 1989 Japan ( edit ) Toyota Store Aichi Toyopet Store , Saitama Corolla Store Kyotanabe Netz Store , Tsurumi - ku , Osaka Wikimedia
Commons has media related to Toyota dealerships . In Japan , Toyota currently maintains separate dealership sales channels . The first sales channel established
in 1946 called `` Toyota Store '' ( トヨタ 店 Toyota Mise ) sells large luxury sedans such as the Toyota Century , and the Toyota Crown . In 1955 , the `` Toyopet
Store '' ( トヨペット 店 Toyopetto - ten ) arrived , originally established to sell the Toyota Corona and the Toyopet ToyoAce truck .
</item 8>

<item 9>
Toyota
Japan : Toyota Motor Corporation . January 29 , 2013 . Retrieved February 3 , 2014 . Jump up ^ `` December 2013 Sales Chart '' ( Press release ) . USA : Toyota
. January 2013 . Retrieved January 10 , 2014 . Jump up ^ `` TMC Announces Results for December 2014 and CY2014 '' . Jump up ^ `` Toyota U.S. Sales Summary '' (
Press release ) . USA : Toyota . Retrieved February 14 , 2015 . Jump up ^ `` TMC Announces Results for December 2015 and CY2015 '' . Jump up ^ `` TMC Announces
Results for December 2016 and CY2016 '' . Jump up ^ Toyota : A history of the First 50 Years . Toyota Motor Corporation . 1988 . ISBN 0 - 517 - 61777 - 3 . ,
p461 . ^ Jump up to : Harden , Blaine ( July 13 , 2008 ) . `` Japan 's Killer Work Ethic '' . Washington Post Foreign Service .
</item 9>

<item 10>
Toyota
North America ( edit ) Main article : Toyota Motor Engineering & Manufacturing North America Toyota Motor North America headquarters is located in Plano , Texas
, and operates as a holding company in North America . Its manufacturing headquarters is located in Erlanger , Kentucky , and is known as Toyota Motor
Engineering & Manufacturing North America . Toyota Canada Inc. has been in production in Canada since 1983 with an aluminium wheel plant in Delta , British
Columbia , which currently employs a workforce of roughly 260 . Its first vehicle assembly plant , in Cambridge , Ontario , since 1988 , now produces Corolla
compact cars , Matrix crossover vehicles , and Lexus RX 350 luxury SUVs , with a workforce of 4,300 workers . Its second assembly operation in Woodstock ,
Ontario , began manufacturing the RAV4 late in 2008 . In 2006 , Toyota 's subsidiary Hino Motors opened a heavy duty truck plant , also in Woodstock , employing
45 people and producing 2000 trucks annually .
</item 10>
=== END ITEM id=aab1fd2536cd111f ===

=== ITEM id=bddcc5ea7636ee11 ===
Gold sentences and facts:
Sentence 0: The Regulatory Reform ( Fire Safety ) Order 2005 ( officially listed as The Regulatory Reform ( Fire Safety ) Order 2005 S.I. 2005 No. 1541 ) is a statutory instrument , applicable only in England and Wales . The Order places the responsibility on individuals within an organisation to carry out risk assessments to identify , manage and reduce the risk of fire . The Order was made into law on 7 June 2005 and came into force on 1 October 2006 .
  Fact 0.0: The Regulatory Reform (Fire Safety) Order 2005 is officially listed as The Regulatory Reform (Fire Safety) Order 2005 S.I. 2005 No. 1541.
  Fact 0.1: The Regulatory Reform (Fire Safety) Order 2005 is a statutory instrument.
  Fact 0.2: The Regulatory Reform (Fire Safety) Order 2005 applies only in England and Wales.
  Fact 0.3: The Regulatory Reform (Fire Safety) Order 2005 places the responsibility on individuals within an organisation to carry out risk assessments.
  Fact 0.4: The risk assessments required by the Regulatory Reform (Fire Safety) Order 2005 identify, manage and reduce the risk of fire.
  Fact 0.5: The Regulatory Reform (Fire Safety) Order 2005 was made into law on 7 June 2005.
  Fact 0.6: The Regulatory Reform (Fire Safety) Order 2005 came into force on 1 October 2006.

Retrieved items:
<item 1>
United Kingdom corporation tax
Finance Acts ( 1965 -- ) , HMSO . From 1998 onwards , consolidation took place , into the Income and Corporation Taxes Act 1988 , the Taxation of Chargeable
Gains Act and the Capital Allowances Act 2001 . `` Reform of corporation tax -- A consultation document '' ( PDF ) . Inland Revenue ( now HMRC ) , and HM
Treasury . August 2002 . Archived from the original ( PDF ) on 6 February 2007 . Retrieved 17 May 2007 . `` Corporation tax reform -- A consultation document ''
( PDF ) . Inland Revenue ( now HMRC ) , and HM Treasury . August 2003 . Archived from the original ( PDF ) on 12 April 2007 . Retrieved 17 May 2007 . ``
Corporation tax reform -- Technical note '' ( PDF ) . Inland Revenue ( now HMRC ) . December 2004 . Archived from the original ( PDF ) on 21 February 2007 .
Retrieved 17 May 2007 . `` Key economic events 1955 -- 1979 '' .
</item 1>

<item 2>
Stand-your-ground law
The law was introduced in response to DPP v. Padraig Nally . A person who uses such force as is permitted by section 2 in the circumstances referred to in that
section shall not be liable in tort with respect to any injury , loss or damage arising from the use of such force . The force used is only such as is
reasonable in the circumstances as he or she believes them to be -- ( i ) to protect himself or herself or another person present in the dwelling from injury ,
assault , detention or death caused by a criminal act , ( ii ) to protect his or her property or the property of another person from appropriation , destruction
or damage caused by a criminal act , or ( iii ) to prevent the commission of a crime or to effect , or assist in effecting , a lawful arrest . It does not
matter whether the person using the force had a safe and practicable opportunity to retreat from the dwelling before using the force concerned . This law does
not apply to force used against a member of An Garda Siochána ( Irish Police ) or anyone assisting them , or a person lawfully performing a function authorised
by or under any enactment .
</item 2>

<item 3>
Graph query row:
entity: Regulatory Reform (Fire Safety) Order 2005
relationship: RELATED_TO
connected_entity: Order came into force
date: 1 October 2006
fact: None
context: None
</item 3>

<item 4>
Regulatory Reform (Fire Safety) Order 2005
Regulatory Reform ( Fire Safety ) Order 2005 - wikipedia Regulatory Reform ( Fire Safety ) Order 2005 Jump to : navigation , search The Regulatory Reform ( Fire
Safety ) Order 2005 ( officially listed as The Regulatory Reform ( Fire Safety ) Order 2005 S.I. 2005 No. 1541 ) is a statutory instrument , applicable only in
England and Wales . The Order places the responsibility on individuals within an organisation to carry out risk assessments to identify , manage and reduce the
risk of fire . The Order was made into law on 7 June 2005 and came into force on 1 October 2006 . Guidance for Businesses and Organisation is available in the
form of 16 Government Published documents , with general guidance , a 5 - Step Checklist and 12 documents pertaining specifically to a particular type of
business premises . It does not normally apply to domestic premises . Contents ( hide ) 1 Guides 2 See also 3 References 4 External links Guides ( edit ) The
guides available are Do you have paying guests ?
</item 4>

<item 5>
Regulatory Reform (Fire Safety) Order 2005
wikipedia.org/w/index.php?title=Regulatory_Reform_(Fire_Safety)_Order_2005&oldid=636696481 '' Categories : Fire and rescue in the United Kingdom Statutory
Instruments of the United Kingdom 2005 in British law Talk Contents About Wikipedia Add links This page was last edited on 5 December 2014 , at 01 : 52 . Text
is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
Central London Property Trust Ltd v High Trees House Ltd
In my opinion , the time has now come for the validity of such a promise to be recognized . The logical consequence , no doubt is that a promise to accept a
smaller sum in discharge of a larger sum , if acted upon , is binding notwithstanding the absence of consideration : and if the fusion of law and equity leads
to this result , so much the better . That aspect was not considered in Foakes v. Beer ( 1884 ) 9 App . Cas. 605 . At this time of day however , when law and
equity have been joined together for over seventy years , principles must be reconsidered in the light of their combined effect . It is to be noticed that in
the Sixth Interim Report of the Law Revision Committee , pars. 35 , 40 , it is recommended that such a promise as that to which I have referred should be
enforceable in law even though no consideration for it has been given by the promisee . It seems to me that , to the extent I have mentioned , that result has
now been achieved by the decisions of the courts .
</item 6>

<item 7>
Indian High Courts Act 1861
2015 National Security Act ( India ) Protection of Women from Domestic Violence Act , 2005 Scheduled Caste and Scheduled Tribe ( Prevention of Atrocities ) Act
, 1989 Education Indian Institutes of Management Act , 2017 Institutes of Technology Act , 1961 National Institutes of Technology Act , 2007 Right of Children
to Free and Compulsory Education Act , 2009 University Grants Commission Act , 1956 Environment Air ( Prevention and Control of Pollution ) Act Biological
Diversity Act , 2002 CAMPA bill Environment Protection Act , 1986 Indian Forest Act , 1927 National Green Tribunal Act Prevention of Cruelty to Animals Act
Protection of Plant Varieties and Farmers ' Rights Act , 2001 Wildlife Protection Act , 1972 Financial Aadhaar ( Targeted Delivery of Financial and other
Subsidies , benefits and services ) Act , 2016 Banking Regulation Act , 1949 Conservation of Foreign Exchange and Prevention of Smuggling Activities Act
Depositories Act The Electricity Act , 2003 Expenditure Tax Act , 1987 Finance Act ( India ) Fiscal Responsibility and Budget Management Act ,
</item 7>

<item 8>
Regulatory Reform (Fire Safety) Order 2005
Fire safety risk assessment : animal premises and stables Fire safety risk assessment : means of escape for disabled people Fire safety risk assessment : open -
air events and venues Fire safety risk assessment : transport premises and facilities Fire safety risk assessment : healthcare premises Fire safety risk
assessment : residential care premises Fire safety risk assessment : theatres , cinemas and similar premises Making your premises safe from fire Fire safety
risk assessment : 5 - step checklist Fire safety risk assessment : factories and warehouses Fire safety risk assessment : large places of assembly Fire safety
risk assessment : small and medium places of assembly Fire safety risk assessment : educational premises Fire safety risk assessment : sleeping accommodation
Fire safety risk assessment : offices and shops See also ( edit ) Fire services in the United Kingdom References ( edit ) Jump up ^ The Regulatory Reform ( Fire
Safety ) Order 2005 , England & Wales Retrieved on 20 March 2014 Jump up ^ The Regulatory Reform ( Fire Safety ) Order 2005 , England & Wales Retrieved on 20
march 2014 Jump up ^ Fire safety law and guidance documents for business Retrieved on 20 march 2014 External links ( edit ) The Regulatory Reform ( Fire Safety
) Order 2005 Retrieved from `` https://en.
</item 8>

<item 9>
Indian High Courts Act 1861
1994 The Prohibition of Child Marriage Act , 2006 The Scheduled Tribes and Other Traditional Forest Dwellers ( Recognition of Forest Rights ) Act , 2006 Street
Vendors Act , 2014 State Laws Anti-Superstition and Black Magic Act The Bombay Prohibition Act , 1949 Bombay Prohibition ( Gujarat Amendment ) 2009 Chhattisgarh
Food Security Act , 2012 Maharashtra Control of Organised Crime Act The Gujarat Local Authorities Laws ( Amendment ) Bill , 2009 Puducherry Prevention of
Anti-Social Activities Act State Reorganisation Andhra Pradesh Reorganisation Act , 2014 Bihar Reorganisation Act , 2000 Madhya Pradesh Reorganisation Act ,
2000 Punjab Reorganisation Act , 1966 States Reorganisation Act , 1956 Uttar Pradesh Reorganisation Act , 2000 Terrorism Unlawful Activities ( Prevention ) Act
Transport Aircraft Act Motor Vehicles Act , 1988 National Waterways Act , 2016 Inland Vessels Act Metro Railways Act , 1978 Metro Railway Act , 2002
Organisation / Body Chartered Accountants Act , 1949 Companies Act 2013 Gram Nyayalayas Act ,
</item 9>

<item 10>
Indian High Courts Act 1861
1987 Finance Act ( India ) Fiscal Responsibility and Budget Management Act , 2003 Foreign Contribution Regulation Act Foreign Exchange Management Act
Geographical Indications of Goods ( Registration and Protection ) Act , 1999 Government Securities Act , 2006 Indian Contract Act , 1872 Indian Stamp Act , 1899
Insolvency and Bankruptcy Code , 2016 Insurance Act , 1938 Negotiable Instruments Act , 1881 Securities Laws ( Amendment ) Act , 2014 Securitisation and
Reconstruction of Financial Assets and Enforcement of Security Interest Act , 2002 The Competition Act , 2002 The High Denomination Bank Notes ( Demonetisation
) Act , 1978 The Income - tax Act , 1961 Transfer of Property Act 1882 Urban Land ( Ceiling and Regulation ) Act , 1976 Healthcare Central Council of
Homoeopathy Act , 1973 Clinical Establishments ( Registration and Regulation ) Act , 2010 Dentist Act Drugs and Cosmetics Act , 1940 Drugs and Magic Remedies (
Objectionable Advertisements ) Act , 1954 Indian Medical Council Act Mental Health Act , 1987 Narcotic Drugs and Psychotropic Substances Act ,
</item 10>
=== END ITEM id=bddcc5ea7636ee11 ===

=== ITEM id=49782a2a4f879ec7 ===
Gold sentences and facts:
Sentence 0: 1924 Winter Olympics medal table Rank NOC Gold Silver Bronze Total Norway ( NOR ) 7 6 17 Finland ( FIN ) 11 Austria ( AUT ) 0 Switzerland ( SUI ) 0 United States ( USA ) 6 Great Britain ( GBR ) 7 Sweden ( SWE ) 0 8 Canada ( CAN ) 0 0 9 France ( FRA ) * 0 0 10 Belgium ( BEL ) 0 0 Total ( 10 NOCs ) 16 16 17 49
  Fact 0.0: The 1924 Winter Olympics medal table has the columns Rank, NOC, Gold, Silver, Bronze, and Total.
  Fact 0.1: The medal table lists Norway (NOR) with the values 7, 6, and 17.
  Fact 0.2: The medal table lists Finland (FIN) with the value 11.
  Fact 0.3: The medal table lists Austria (AUT) with the value 0.
  Fact 0.4: The medal table lists Switzerland (SUI) with the value 0.
  Fact 0.5: The medal table lists United States (USA) with the value 6.
  Fact 0.6: The medal table lists Great Britain (GBR) with the value 7.
  Fact 0.7: The medal table lists Sweden (SWE) with the values 0 and 8.
  Fact 0.8: The medal table lists Canada (CAN) with the values 0, 0, and 9.
  Fact 0.9: The medal table lists France (FRA), marked with an asterisk, with the values 0, 0, and 10.
  Fact 0.10: The medal table lists Belgium (BEL) with the values 0 and 0.
  Fact 0.11: The medal table's total row for the 10 NOCs lists the values 16, 16, 17, and 49.
Sentence 1: The 1924 Winter Olympics , officially known as the I Olympic Winter Games , and known at the time as Semaine Internationale des Sports d'Hiver ( `` International Winter Sports Week '' ) , was a winter multi-sport event held in Chamonix , France , from 25 January to 5 February 1924 . Norway topped the table , collecting seventeen medals in total , including four gold , three of which were won by Thorleif Haug in the nordic combined and cross-country skiing events . Norway also achieved two podium sweeps , winning all three medals in both the 50 km cross-country skiing and the nordic combined . This remained a record at the Winter Olympics until 2014 .
  Fact 1.0: The 1924 Winter Olympics was officially known as the I Olympic Winter Games.
  Fact 1.1: At the time, the 1924 Winter Olympics was known as Semaine Internationale des Sports d'Hiver ("International Winter Sports Week").
  Fact 1.2: The 1924 Winter Olympics was a winter multi-sport event.
  Fact 1.3: The 1924 Winter Olympics was held in Chamonix, France.
  Fact 1.4: The 1924 Winter Olympics was held from 25 January to 5 February 1924.
  Fact 1.5: Norway topped the medal table of the 1924 Winter Olympics.
  Fact 1.6: Norway collected seventeen medals in total at the 1924 Winter Olympics.
  Fact 1.7: Norway collected four gold medals at the 1924 Winter Olympics.
  Fact 1.8: Three of Norway's four gold medals at the 1924 Winter Olympics were won by Thorleif Haug.
  Fact 1.9: Thorleif Haug won his gold medals in the nordic combined and cross-country skiing events.
  Fact 1.10: Norway achieved two podium sweeps at the 1924 Winter Olympics.
  Fact 1.11: Norway won all three medals in the 50 km cross-country skiing event at the 1924 Winter Olympics.
  Fact 1.12: Norway won all three medals in the nordic combined event at the 1924 Winter Olympics.
  Fact 1.13: Norway's two podium sweeps at the 1924 Winter Olympics remained a record at the Winter Olympics until 2014.

Retrieved items:
<item 1>
1924 Winter Olympics medal table
Two bronze medals were awarded in the 500 metres speed skating event for the third place tie . Key * Host nation ( France ) 1924 Winter Olympics medal table
Rank NOC Gold Silver Bronze Total Norway ( NOR ) 7 6 17 Finland ( FIN ) 11 Austria ( AUT ) 0 Switzerland ( SUI ) 0 United States ( USA ) 6 Great Britain ( GBR )
7 Sweden ( SWE ) 0 8 Canada ( CAN ) 0 0 9 France ( FRA ) * 0 0 10 Belgium ( BEL ) 0 0 Total ( 10 NOCs ) 16 16 17 49 See also ( edit ) List of 1924 Winter
Olympics medal winners Notes ( edit ) Jump up ^ No country achieved more than two podium sweeps until the Netherlands in 2014 , but the record was tied prior to
that ; by Norway themselves in 1928 and 1932 , the Soviet Union in 1964 , East Germany in 1972 and 1984 , and Germany in 1998 . References ( edit ) ^ Jump up to
: `` 1924 Medal Tally '' .
</item 1>

<item 2>
1924 Winter Olympics medal table
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 2>

<item 3>
1924 Winter Olympics medal table
Partly due to this , varying figures are reported for the number of participants . The International Olympic Committee ( IOC ) website suggests that 258
athletes from 16 nations participated in 16 events across 9 sports . In contrast , the Sports - Reference website lists 313 participants from 19 countries .
Bill Mallon , a prominent Olympic Games historian , quotes a figure of 291 competitors in his Historical Dictionary of the Olympic Movement . Meanwhile the
official report for the 1924 Summer and Winter Olympics listed 293 athletes from 17 nations . Finland placed second on the table , collecting four gold medals
amongst a total of eleven . Clas Thunberg won five of their medals , achieving a medal in each of the speed skating events he took part in ; collecting three
gold medals , one silver and one bronze . Eight of the competing nations achieved at least one gold medal , with only Belgium and the hosts , France , medalling
without winning a gold medal . One medal was reallocated in 1974 ; during the games , Thorleif Haug of Norway had been awarded the bronze medal in ski jumping ,
along with his three gold medals .
</item 3>

<item 4>
1924 Winter Olympics medal table
1924 Winter Olympics medal table - wikipedia 1924 Winter Olympics medal table Jump to : navigation , search Publicity poster for the event The 1924 Winter
Olympics , officially known as the I Olympic Winter Games , and known at the time as Semaine Internationale des Sports d'Hiver ( `` International Winter Sports
Week '' ) , was a winter multi-sport event held in Chamonix , France , from 25 January to 5 February 1924 . Norway topped the table , collecting seventeen
medals in total , including four gold , three of which were won by Thorleif Haug in the nordic combined and cross-country skiing events . Norway also achieved
two podium sweeps , winning all three medals in both the 50 km cross-country skiing and the nordic combined . This remained a record at the Winter Olympics
until 2014 . When it was held , the games were not formally recognised as being the Olympics , but acknowledged that they were held under the `` high patronage
of the International Olympic Committee '' . Partly due to this , varying figures are reported for the number of participants .
</item 4>

<item 5>
1924 Winter Olympics medal table
The number of silver medals is taken into consideration next and then the number of bronze medals . If nations are still tied , equal ranking is given and they
are listed alphabetically . Two bronze medals were awarded in the 500 metres speed skating event for the third place tie .
</item 5>

<item 6>
1924 Winter Olympics medal table
wikipedia.org/w/index.php?title=1924_Winter_Olympics_medal_table&oldid=824811466 '' Categories : 1924 Winter Olympics Winter Olympics medal tables Hidden
categories : CS1 French - language sources ( fr ) CS1 Norwegian - language sources ( no ) Use dmy dates from October 2017 Use British English from October 2017
Featured lists Talk Contents About Wikipedia Беларуская Български Català Čeština Deutsch Español Esperanto فارسی Français 한국어 हिन्दी Magyar Македонски
Nederlands 日本 語 Polski Português Русский Slovenščina Suomi Svenska ไทย Türkçe Українська اردو 中文 17 more Edit links This page was last edited on 9 February 2018
, at 17 : 05 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to
the Terms of Use and Privacy Policy .
</item 6>

<item 7>
1924 Winter Olympics medal table
Sports Reference . Archived from the original on 25 August 2017 . Retrieved 25 August 2017 . Jump up ^ Mallon , Bill ; Heijmans , Jereon ( 2011 ) . Historical
Dictionary of the Olympic Movement . Plymouth , UK : Scarecrow Press . p. 46 . ISBN 978 - 0 - 8108 - 7522 - 7 . Archived from the original on 14 October 2017 .
Jump up ^ `` Official Report of the 1924 Olympic Games ( PDF ) ( in French ) . International Olympic Committee . 1924 . p. 662 . Archived ( PDF ) from the
original on 26 February 2016 . Jump up ^ Jorsett , Per ( 13 February 2009 ) . `` Thorleif Haug '' . Norsk biografisk leksikon ( in Norwegian ) . Oslo :
Kunnskapsforlaget . Archived from the original on 21 September 2017 . Retrieved 20 September 2017 .
</item 7>

<item 8>
1924 Winter Olympics medal table
International Olympic Committee . Archived from the original on 4 December 2008 . Retrieved 22 October 2017 . ( hide ) Olympic Games medal tables Olympic medal
All - time Olympic Games medal table Pierre de Coubertin medal Intercalated Games ( 1906 ) Summer Olympic Games 1896 1900 1904 1908 1912 1920 1924 1928 1932
1936 1948 1952 1956 1960 1964 1968 1972 1976 1980 1984 1988 1992 2000 2008 2012 2016 Winter Olympic Games 1924 1928 1932 1936 1948 1952 1956 1960 1964 1968 1972
1976 1980 1984 1988 1992 1998 2002 2006 2014 2018 Lists of Olympic medalists List of stripped Olympic medals List of Olympic medals by host nation List of ties
for medals at the Olympics Retrieved from `` https://en.wikipedia.
</item 8>

<item 9>
1924 Winter Olympics medal table
A Norwegian historian discovered that there had been a scoring error , and an American , Anders Haugen , had actually finished in third place . As Haug had died
in 1934 , his daughter presented Haugen , aged 83 , with his medal . In the official report , the classification of nations is ranked not by medals , but rather
by `` points '' , which were awarded for each position from first to sixth ; not exclusively for gold , silver and bronze . Due to this , Czechoslovakia were
included in the table ahead of Belgium , due to two fourth - place finishes , one fifth and one sixth . Italy were also present , in twelfth , for their
solitary sixth place . Contents ( hide ) 1 Medal table 2 See also 3 Notes 4 References Medal table ( edit ) Clas Thunberg won five speed skating medals at the
games for Finland . The medal table is based on information provided by the IOC and is consistent with IOC convention in its published medal tables . By default
, the table is ordered by the number of gold medals the athletes from a nation have won ( in this context , a nation is an entity represented by a National
Olympic Committee ) .
</item 9>

<item 10>
1924 Winter Olympics medal table
References ( edit ) ^ Jump up to : `` 1924 Medal Tally '' . Topend Sports . Archived from the original on 25 August 2017 . Retrieved 25 August 2017 . ^ Jump up
to : `` Chamonix 1924 '' . International Olympic Committee . Archived from the original on 1 September 2017 . Retrieved 24 August 2017 . Jump up ^ BMallon ( 18
February 2014 ) . `` Netherlands Speed Skating Medal Sweeps '' . OlympStats . Archived from the original on 25 August 2017 . Retrieved 25 August 2017 . Jump up
^ `` Bergsma breaks Olympic record to lead fourth Dutch medal sweep '' . Xinhuanet. 19 February 2014 . Archived from the original on 23 September 2015 .
Retrieved 25 August 2017 . ^ Jump up to : `` 1924 Chamonix Winter Games '' . Sports Reference . Archived from the original on 25 August 2017 .
</item 10>
=== END ITEM id=49782a2a4f879ec7 ===

=== ITEM id=89b133cdb4a16977 ===
Gold sentences and facts:
Sentence 0: Capital punishment in Australia has been abolished in all jurisdictions . Queensland abolished the death penalty in 1922 . Tasmania did the same in 1968 , the federal government abolished the death penalty in 1973 , with application also in the Australian Capital Territory and the Northern Territory . Victoria did so in 1975 , South Australia in 1976 , and Western Australia in 1984 . New South Wales abolished the death penalty for murder in 1955 , and for all crimes in 1985 . In 2010 , the federal government passed legislation prohibiting the re-establishment of capital punishment by any state or territory . Neither the Commonwealth nor any of the states will extradite or deport a prisoner to another jurisdiction if they will face the death penalty , and police co-operation with other countries which have the death penalty has been questioned .
  Fact 0.0: Capital punishment in Australia has been abolished in all jurisdictions.
  Fact 0.1: Queensland abolished the death penalty in 1922.
  Fact 0.2: Tasmania abolished the death penalty in 1968.
  Fact 0.3: The federal government abolished the death penalty in 1973.
  Fact 0.4: The federal abolition of the death penalty in 1973 applied in the Australian Capital Territory.
  Fact 0.5: The federal abolition of the death penalty in 1973 applied in the Northern Territory.
  Fact 0.6: Victoria abolished the death penalty in 1975.
  Fact 0.7: South Australia abolished the death penalty in 1976.
  Fact 0.8: Western Australia abolished the death penalty in 1984.
  Fact 0.9: New South Wales abolished the death penalty for murder in 1955.
  Fact 0.10: New South Wales abolished the death penalty for all crimes in 1985.
  Fact 0.11: In 2010, the federal government passed legislation prohibiting the re-establishment of capital punishment by any state or territory.
  Fact 0.12: The Commonwealth will not extradite a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.13: No state will extradite a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.14: The Commonwealth will not deport a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.15: No state will deport a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.16: Police co-operation with other countries which have the death penalty has been questioned.

Retrieved items:
<item 1>
Capital punishment in Australia
During the 19th century , these crimes saw about 80 people hanged each year throughout Australia . Before and after federation , each state made its own
criminal laws and punishments . For a full list see : Main article : List of people legally executed in Australia Commonwealth ( edit ) In 1973 the Death
Penalty Abolition Act 1973 of the Commonwealth abolished the death penalty for federal offences . It provided in Section 3 that the Act applied to any offence
against a law of the Commonwealth , the Territories or under an Imperial Act , and in s . 4 that `` ( a ) person is not liable to the punishment of death for
any offence '' . No executions were carried out under the bridge of the federal government and the passage of the Death Penalty Abolition Act 1973 saw the death
penalty replaced with life imprisonment as their maximum punishment . Since the Commonwealth effects of utilising this Act no more individuals have been exposed
to the death penalty and it is now replaced with life imprisonment . On 11 March 2010 Federal Parliament passed laws that prevent the death penalty from being
reintroduced by any state or territory in Australia .
</item 1>

<item 2>
Capital punishment in Australia
Northern Territory ( edit ) Aborigines who lived in the Northern Territory came under the European law of Australia even though they did not have any contact
with the government . There were several outcries over Aborigines receiving mandatory death sentences for murder , leading to the passage of the Crimes
Ordinance 1934 which allowed for discretionary sentences when both the accused and the victim were Aboriginal . The last execution was a double hanging in 1952
, and the death penalty was abolished 1973 . Public opinion ( edit ) In Australia capital punishment was banned on a state - by - state basis through the 20th
century . Despite the ban , polls have indicated varying support for the reintroduction of the practice . A 2005 Bulletin poll showed that most Australians
supported capital punishment . The Australian National University 's 2007 Electoral Survey found that 44 per cent of people thought the death penalty should be
reintroduced , while only 38 per cent disagreed . In the recent case of the Bali bombers , then prime minister John Howard stated that Australians expected
their execution by Indonesia . On occasion the issue of capital punishment is published in the media or is subject to media and public support and scrutiny .
</item 2>

<item 3>
Capital punishment in Australia
The last death sentence was given in August 1984 , when Brenda Hodge was sentenced to death by Western Australia ( and subsequently had her sentence commuted to
life imprisonment ) . Contents ( hide ) 1 History 1.1 Commonwealth 1.2 New South Wales 1.3 Victoria 1.4 Queensland 1.5 Western Australia 1.6 South Australia 1.7
Tasmania 1.8 Australian Capital Territory 1.9 Northern Territory 2 Public opinion 3 References 4 External links History ( edit ) Death sentences were carried
out under Aboriginal customary law , either directly or through sorcery . In some cases the condemned could be denied mortuary rites . The first executions
carried out under European law in Australia took place in Western Australia in 1629 , when Dutch authorities hanged the mutineers of the Batavia . Capital
punishment had been part of the legal system of Australia since British settlement and during the 19th century , crimes that could carry a death sentence
included burglary , sheep stealing , forgery , sexual assaults , murder and manslaughter , and there is one reported case of someone being executed for `` being
illegally at large '' .
</item 3>

<item 4>
Capital punishment in Australia
Capital punishment in Australia - wikipedia Capital punishment in Australia Jump to : navigation , search Capital punishment in Australia has been abolished in
all jurisdictions . Queensland abolished the death penalty in 1922 . Tasmania did the same in 1968 , the federal government abolished the death penalty in 1973
, with application also in the Australian Capital Territory and the Northern Territory . Victoria did so in 1975 , South Australia in 1976 , and Western
Australia in 1984 . New South Wales abolished the death penalty for murder in 1955 , and for all crimes in 1985 . In 2010 , the federal government passed
legislation prohibiting the re-establishment of capital punishment by any state or territory . Neither the Commonwealth nor any of the states will extradite or
deport a prisoner to another jurisdiction if they will face the death penalty , and police co-operation with other countries which have the death penalty has
been questioned . The last execution in Australia took place in 1967 , when Ronald Ryan was hanged in Victoria . Between Ryan 's execution in 1967 and 1984 ,
several more people were sentenced to death , but had their sentences commuted to life imprisonment .
</item 4>

<item 5>
Capital punishment in Australia
New South Wales was the last Australian State to formally abolish the death penalty for all crimes . Victoria ( edit ) Old Melbourne Gaol gallows Victoria 's
first executions occurred in 1842 when two Aboriginal men , Tunnerminnerwait and Maulboyheenner , were hanged outside the site of the Melbourne Gaol for the
killing of two whalers in the Westernport district . Ronald Ryan was the last man executed at Pentridge Prison and in Australia . Ryan was hanged on 3 February
1967 after being convicted of the shooting death of a prison officer during a prison escape from Pentridge Prison , Coburg , Victoria in 1965 . Ryan was the
last of 186 executions . Victoria was also the state of the last woman executed in Australia . Jean Lee was hanged in 1951 . She was accused of being an
accomplice in the murder of 73 - year - old William ( ' Pop ' ) Kent . She , along with her accomplices were executed on 19 February 1951 .
</item 5>

<item 6>
Capital punishment in Australia
Retrieved 24 April 2017 . Jump up ^ `` Small majority of Australians favour the death penalty for deadly terrorist acts in Australia '' . Roy Morgan . Melbourne
, Vic. : Roy Morgan Research Ltd. 19 September 2014 . Retrieved 3 December 2017 . Sources ^ Death Penalty Abolition Act 1973 Capital Punishment from the
Fremantle Prison site Trends & Issues in Crime and Criminal Justice No. 3 : Capital punishment from the Australian Institute of Criminology Dhakiyarr Wirrpanda
-- Appeal for Justice `` Capital Punishment '' speech by The Ian Callinan , Justice of the High Court of Australia The History of Correctional Services in South
Australia External links ( edit ) `` List of Executions in the State of Victoria '' . Archived from the original on 8 October 2011 . `` The Death Penalty (
Queensland ) '' . Life & Death in the Sunshine State . Archived from the original on 11 May 2015 .
</item 6>

<item 7>
Capital punishment in Australia
Jump up ^ Helen MacDonald ( 2005 ) . Human remains : Dissection and Its Histories . Yale University Press . ISBN 978 - 0 - 300 - 11699 - 1 . Jump up ^ `` Two
executed '' . Northern Standard . Darwin , NT. 8 August 1952 . p. 1 . Retrieved 3 December 2017 . Jump up ^ Williams , George ( 2 March 2010 ) . `` No death
penalty , no shades of grey '' . Sydney Morning Herald . Retrieved 3 December 2017 . Jump up ^ Australians expect Bali bomber executions , says PM By Sarah
Smiles Canberra and Mark Forbes , 13 October 2007 , Brisbane Times Jump up ^ Death penalty dead and buried as Parliament bans it for good . Herald Sun ( 2010 -
03 - 12 ) . Jump up ^ `` Australians say penalty for murder should be Imprisonment ( 64 % ) rather than the Death Penalty ( 23 % ) '' , Roy Morgan Research , 27
August 2009 . Retrieved 24 April 2017 .
</item 7>

<item 8>
Capital punishment in Australia
Australian Federal Police . Retrieved 3 December 2017 . Jump up ^ Finlay , Lorraine ( March 2011 ) . `` Exporting the Death Penalty ? Reconciling International
Police Cooperation and the Abolition of the Death Penalty in Australia '' ( pdf ) . Sydney Law Review . 33 ( 1 ) : 95 -- 117 . ISSN 0082 - 0512 . Retrieved 3
December 2017 . Jump up ^ Traditional Aboriginal Law and Punishment Archived 6 March 2009 at the Wayback Machine . -- Part V -- Aboriginal Customary Law and the
Criminal Justice System , Law Reform Commission of Western Australia -- Aboriginal Customary Laws Discussion Paper Jump up ^ `` British Justice '' . Museum
Victoria Australia . Retrieved 14 March 2017 . Jump up ^ The Hanging of Ronald Ryan : 40 Years Later Archived 28 December 2007 at the Wayback Machine ...
Cosmos.bcst.yahoo.com ( 2009 - 08 - 27 ) . Retrieved on 2012 - 09 - 10 . Jump up ^ Biography -- Jean Lee -- Australian Dictionary of Biography .
</item 8>

<item 9>
Capital punishment in Australia
The last sentence of death in Western Australia ( indeed anywhere in Australia ) was passed in 1984 , but the female killer ( Brenda Hodge ) in question had her
sentence commuted to imprisonment for life , as was customary by this stage . Capital punishment was formally removed from the statutes of the state with the
passage of the Acts Amendment ( Abolition of Capital Punishment ) Act 1984 . South Australia ( edit ) The Adelaide Gaol was the site of forty - four hangings ,
from Joseph Stagg on 18 November 1840 to Glen Sabre Valance , murderer and rapist , on 24 November 1964 and three executions also occurred at Mount Gambier Gaol
. Two Ngarrindjeri men were controversially executed by hanging along the Coorong on 22 August 1840 , after a drumhead court - martial conducted by Police
Commissioner O'Halloran on the orders of Governor George Gawler . The men were found to be guilty of murdering the twenty - five survivors of the shipwreck
Maria a few months before .
</item 9>

<item 10>
Capital punishment in Australia
Elizabeth Woolcock , the only woman ever to have been executed under South Australian law , was hanged on 30 December 1873 . Her body was not released to the
family and was buried between the inner and outer walls of the prison , identified by a number and the date of the execution . In 1976 , the Criminal Law
Consolidation Act was modified so that the death sentence was changed to life imprisonment . Tasmania ( edit ) In the early days of colonial rule Tasmania ,
then known as Van Diemen 's Land , was the site of penal transports . Mary McLauchlan was convicted in 1830 for infanticide ; she was sentenced to both death
and dissection . She was the first woman to be hanged in Tasmania . The last execution was in 1946 , that of serial murderer and rapist Frederick Thompson . The
death penalty was abolished in 1968 . Australian capital Territory ( edit ) No executions were carried out in the Australian Capital Territory , where federal
legislation abolished capital punishment in 1973 .
</item 10>
=== END ITEM id=89b133cdb4a16977 ===

=== ITEM id=28b0e41e299c1807 ===
Gold sentences and facts:
Sentence 0: With the Intertropical Convergence Zone ( ITCZ ) swinging northward over West Africa from the Southern Hemisphere in April , heavy showers coming from pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria known as the Maritime tropical ( south westerlies ) and the Continental tropical ( north easterlies ) , begins in central Nigeria while the Monsoons from the south atlantic ocean arrives in central Nigeria in July bringing with it high humidity , heavy cloud cover and heavy rainfall which can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in central Nigeria varies from 1,100 mm ( 43.3 in ) in the lowlands of the river Niger Benue trough to over 2,000 mm ( 78.7 in ) along the south western escarpment of the Jos Plateau .
  Fact 0.0: The Intertropical Convergence Zone (ITCZ) swings northward over West Africa from the Southern Hemisphere in April.
  Fact 0.1: In April, heavy showers come from pre-monsoonal convective clouds, mainly in the form of squall lines.
  Fact 0.2: The squall lines are also known as the north easterlies.
  Fact 0.3: The north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria.
  Fact 0.4: The two dominant airmasses in Nigeria are the Maritime tropical (south westerlies) and the Continental tropical (north easterlies).
  Fact 0.5: The pre-monsoonal showers begin in central Nigeria.
  Fact 0.6: The monsoons from the South Atlantic Ocean arrive in central Nigeria in July.
  Fact 0.7: The monsoons bring high humidity, heavy cloud cover and heavy rainfall to central Nigeria in July.
  Fact 0.8: The rainfall from the monsoons can be a daily occurrence in central Nigeria.
  Fact 0.9: The monsoon rainfall in central Nigeria lasts until September.
  Fact 0.10: The monsoons gradually begin retreating southward to the southern part of Nigeria when September arrives.
  Fact 0.11: Rainfall totals in central Nigeria vary from 1,100 mm (43.3 in) in the lowlands of the river Niger Benue trough.
  Fact 0.12: Rainfall totals in central Nigeria reach over 2,000 mm (78.7 in) along the south western escarpment of the Jos Plateau.

Retrieved items:
<item 1>
Geography of Nigeria
In the rainy season it is damp , and the rainfalls are usually abundant . The dry season of Nigeria is a period of little cloud cover in the southern part of
Nigeria to virtually no cloud cover in the northern part of Nigeria . The sun shines through the atmosphere with little obstructions from the clear skies making
the dry season in Nigeria a period of warm weather conditions.In the middle of the dry season around December , a dusty wind from the Sahara Desert called the
harmattan enters Nigeria from the northeastern part of the country blocking sun rays partially from shining and also creating haze in the atmosphere , this
activities of the wind lowers temperatures considerably saving inhabitants for sometime , from the scorching heat that would have occurred as a result of
clearer skies during the dry season . But with the withdrawal of this wind around March to April following the onset of the rainy season , temperatures can go
as high as 44 ° C ( 111.2 ° F ) in some parts of Nigeria . Semi temperate weather conditions prevail on the highlands in central Nigeria above 1,200 metres (
3,937 ft ) above sea level , namely the Jos Plateau .
</item 1>

<item 2>
Geography of Nigeria
These are the dry season and the rainy season . The dry season is accompanied by a dust laden airmass from the Sahara Desert , locally known as Harmattan , or
by its main name , The Tropical Continental ( CT ) airmass , while the rainy season is heavily influenced by an airmass originating from the south atlantic
ocean , locally known as the south west wind , or by its main name , The Tropical Maritime ( MT ) airmass . These two major wind systems in Nigeria are known as
the trade winds . Trade winds ( edit ) Tropical Maritime airmass ( edit ) The tropical maritime airmass ( MT ) is responsible for Nigeria 's rainy season . This
wind ( the tropical maritime airmass ) invades the country from February in the southern part of Nigeria while it takes longer for the wind to fully cover the
whole of the country , reaching the northern part of Nigeria in June . Its invasion is as a result of the northward retreat , of the tropical continental
airmass ( CT ) known as the harmattan .
</item 2>

<item 3>
Geography of Nigeria
This climate , the tropical savanna climate exhibits a well marked rainy season and a dry season with a single peak known as the summer maximum due to its
distance from the equator . Temperatures are above 18 ° C ( 64 ° F ) throughout the year . Abuja , Nigeria 's capital city found in central Nigeria , has a
temperature range of 18.45 ° C ( 65.21 ° F ) to 36.9 ° C ( 98.4 ° F ) , and an annual rainfall of about 1,500 mm ( 59.1 in ) with a single rainfall maxima in
September . The single Dry season experienced in this climate , the tropical savanna climate in central Nigeria beginning from December to march , is hot and
dry with the Harmattan wind , a continental tropical ( CT ) airmass laden with dust from the Sahara Desert prevailing throughout this period .
</item 3>

<item 4>
Geography of Nigeria
With the Intertropical Convergence Zone ( ITCZ ) swinging northward over West Africa from the Southern Hemisphere in April , heavy showers coming from
pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two
dominant airmasses in Nigeria known as the Maritime tropical ( south westerlies ) and the Continental tropical ( north easterlies ) , begins in central Nigeria
while the Monsoons from the south atlantic ocean arrives in central Nigeria in July bringing with it high humidity , heavy cloud cover and heavy rainfall which
can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in
central Nigeria varies from 1,100 mm ( 43.3 in ) in the lowlands of the river Niger Benue trough to over 2,000 mm ( 78.7 in ) along the south western escarpment
of the Jos Plateau . The Sahel climate or tropical dry climate , is the predominant climate type in the northern part of Nigeria . Annual rainfall totals are
lower compared to the southern and central part of Nigeria .
</item 4>

<item 5>
Geography of Nigeria
The Tropical Maritime Airmass is a warm humid and unstable trade wind due to its warmth . Convectional currents are easily set up within the airmass whenever
there is little instability in the airmass as a result of a slight to a very high orographic uplift in mountainous regions like the obudu plateau or the heating
of the land which can trigger the formation of cumulonimbus cloud leading to thunderstorms within the airmass . During the dominance of the Tropical Maritime
Airmass ( MT ) in the rainy season of Nigeria , mornings are bright and sunny , the sun 's heating of the land in the mornings and afternoons sets up
convectional currents , these currents rise vertically and cumulonimbus clouds are formed , by afternoons to evenings , torrential downpour follows . The
Easterly wave or the African easterly waves is another major contributor of rainfall during the summer monsoons months of May to September in Nigeria . The
nature of this waves changes at about the 15 degrees line . The waves that pass south of this line carry moisture and create convection that leads to rain
clouds .
</item 5>

<item 6>
Geography of Nigeria
The rest of the southeast receives between 2,000 and 3,000 mm ( 118.1 in ) of rain per year . The southern region of Nigeria experiences a double rainfall
maxima characterised by two high rainfall peaks , with a short dry season and a longer dry season falling between and after each peaks . The first rainy season
begins around March and last to the end of July with a peak in June , this rainy season is followed by a short dry break in August known as the August break
which is a short dry season lasting for two to three weeks in August . This break is broken by the short rainy season starting around early September and
lasting to Mid October with a peak period at the end of September . The ending of the short rainy season in October is followed by long dry season . This period
starts from late October and lasts till early March with peak dry conditions between early December and late February . The tropical savanna climate or tropical
wet and dry climate , is extensive in area and covers most of Western Nigeria to central Nigeria beginning from the Tropical rainforest climate boundary in
southern Nigeria to the central part of Nigeria , where it exerts enormous influence on the region .
</item 6>

<item 7>
Geography of Nigeria
The northward retreat of the tropical continental airmass ( CT ) , is caused by the sun 's northward shift from the tropic of capricorn in the southern
hemisphere to the tropic of cancer in the northern hemisphere . This shift begins from February and ends in June , when the sun is fully overhead , at the
tropic of cancer in the northern hemisphere . During this northward migration of the sun as a result of the earth tilting along its axis , the sun crosses the
equator ( around March ) , moving over west Africa at this time on its journey to the northern hemisphere . West Africa comes directly under the sun at this
time . The sun is overhead throughout west Africa and over Nigeria during this period of the sun 's northward migration to the tropic of cancer in the northern
hemisphere . The whole of West Africa is heated intensely as result of the increased insolation received from the sun being overhead over west Africa .
Temperatures can climb as high as 35 ° C ( 95.0 ° F ) over west Africa during this time .
</item 7>

<item 8>
Geography of Nigeria
The Tropical monsoon climate has a very small temperature range . Then temperature ranges are almost constant throughout the year , for example , Warri town in
the southern part of Nigeria , records a maximum of 28 ° C ( 82.4 ° F ) for its hottest month while its lowest temperature is 26 ° C ( 78.8 ° F ) in its coldest
month . The temperature difference of Warri town is not more than 2 ° C ( 5 ° F ) . The southern part of Nigeria experiences heavy and abundant rainfall . These
storms are usually convectional in nature due to the regions proximity , to the equatorial belt . The annual rainfall received in this region is very high ,
usually above the 2,000 mm ( 78.7 in ) rainfall totals giving for tropical rainforest climates worldwide . Over 4,000 mm ( 157.5 in ) of rainfall is received in
the coastal region of Nigeria around the Niger delta area . Bonny town found in the coastal region of the Niger delta area in southern Nigeria receives well
over 4,000 mm ( 157.5 in ) of rainfall annually .
</item 8>

<item 9>
Geography of Nigeria
Temperatures in the northern part of Nigeria can go as high as 48 ° C ( 118.4 ° F ) in cities like Maiduguri . The high temperatures coupled with an increase in
insolation causes a region of low pressure to develop over west Africa and Nigeria ( between March to May ) . The Tropical continental airmass ( CT ) from the
Sahara Desert in the northern part of West Africa , is weakened due to the overheating of the land surface in west Africa and Nigeria at this time . The
Tropical continental airmass ( CT ) begins to retreat northwards to the Sahara Desert due to massive heating of the land which transfers heat in the form of
convection into the Tropical continental airmass ( CT ) which constitutes the main layer of air above the land . This transfer of heat in the Tropical
continental airmass ( CT ) in turn , causes the wind to expand and become lighter as this is the normal behaviour for winds moving above intensely heated
grounds .
</item 9>

<item 10>
Geography of Nigeria
The Tropical continental airmass ( CT ) loses its strength as a major airmass in the region of west Africa and over Nigeria at this time ( around February in
the southern part of Nigeria to June in northern Nigeria ) and begins to retreat coupled with the rising of air in form of convection within this airmass (
Tropical continental airmass ( CT ) ) , further weakening the dominance of the wind over west Africa and Nigeria . The Tropical continental airmass ( CT )
finally retreats from most part of Nigeria , and the West African atmosphere around April to May , leaving an empty atmosphere over Nigeria . The sun 's rays
enters into the atmosphere of Nigeria more intense than it does during the presence of the Tropical continental airmass , which contained dust ( in form of haze
) that reduced the intensity of the sun . The overheating of the west Africa land mass and Nigeria in particular creates a low pressure region over west Africa
and Nigeria . This low pressure zone attracts the Tropical Maritime Airmass ( MT ) from the south Atlantic Ocean since areas of low pressures experiences inward
blowing winds because winds are moving air blowing outwards from regions of high pressure to regions of low pressure .
</item 10>
=== END ITEM id=28b0e41e299c1807 ===

=== ITEM id=58f1061afbb16c60 ===
Gold sentences and facts:
Sentence 0: Somatic motor neurons originate in the central nervous system , project their axons to skeletal muscles ( such as the muscles of the limbs , abdominal , and intercostal muscles ) , which are involved in locomotion . The three types of these neurons are the alpha efferent neurons , beta efferent neurons , and gamma efferent neurons . They are called efferent to indicate the flow of information from the central nervous system ( CNS ) to the periphery .
  Fact 0.0: Somatic motor neurons originate in the central nervous system.
  Fact 0.1: Somatic motor neurons project their axons to skeletal muscles.
  Fact 0.2: Examples of those skeletal muscles are the muscles of the limbs, abdominal muscles, and intercostal muscles.
  Fact 0.3: The skeletal muscles that somatic motor neurons project to are involved in locomotion.
  Fact 0.4: There are three types of somatic motor neurons.
  Fact 0.5: The three types are alpha efferent neurons, beta efferent neurons, and gamma efferent neurons.
  Fact 0.6: Somatic motor neurons are called efferent.
  Fact 0.7: The name efferent indicates the flow of information from the central nervous system (CNS) to the periphery.
Sentence 1: The term ' motor neuron ' is usually restricted to the lower motor neurons , the efferent nerves that directly innervate muscles .
  Fact 1.0: The term 'motor neuron' is usually restricted to the lower motor neurons.
  Fact 1.1: Lower motor neurons are the efferent nerves that directly innervate muscles.

Retrieved items:
<item 1>
Motor neuron
Muscle relaxation and inhibition of muscle contraction in vertebrates is obtained only by inhibition of the motor neuron itself . Muscle innervation may
eventually play a role in the maturation of motor activity . This is how muscle relaxants work by acting on the motor neurons that innervate muscles ( by
decreasing their electrophysiological activity ) or on cholinergic neuromuscular junctions , rather than on the muscles themselves . Special visceral motor
neurons ( edit ) These are also known as branchial motor neurons , which are involved in facial expression , mastication , phonation , and swallowing .
Associated cranial nerves are the oculomotor , abducens , trochlear , and hypoglossal nerves .
</item 1>

<item 2>
Motor neuron
Motor neuron - wikipedia Motor neuron Jump to : navigation , search Motor neuron Micrograph of the hypoglossal nucleus showing motor neurons with their
characteristic coarse Nissl substance ( `` tigroid '' cytoplasm ) . H&E - LFB stain . Details Location Ventral horn of the spinal cord , some cranial nerve
nuclei Shape Projection neuron Function Excitatory projection ( to NMJ ) Neurotransmitter UMN to LMN : glutamate ; LMN to NMJ : ACh Presynaptic connections
Primary motor cortex via the Corticospinal tract Postsynaptic connections Muscle fibers and other neurons Identifiers MeSH D009046 NeuroLex ID nifext_103 TA
A14. 2.00. 021 FMA 5867 Anatomical terms of neuroanatomy ( edit on Wikidata ) A motor neuron ( or motoneuron ) is a neuron whose cell body is located in the
motor cortex , brainstem or the spinal cord , and whose axon ( fiber ) projects to the spinal cord or outside of the spinal cord to directly or indirectly
control effector organs , mainly muscles and glands . There are two types of motor neuron -- upper motor neurons and lower motor neurons .
</item 2>

<item 3>
Motor neuron
There are three primary categories lower motor neurons , which can be further divided in sub-categories . According to their targets , motor neurons are
classified into three broad categories : Somatic motor neurons Special visceral motor neurons General visceral motor neurons Somatic motor neurons ( edit )
Somatic motor neurons originate in the central nervous system , project their axons to skeletal muscles ( such as the muscles of the limbs , abdominal , and
intercostal muscles ) , which are involved in locomotion . The three types of these neurons are the alpha efferent neurons , beta efferent neurons , and gamma
efferent neurons . They are called efferent to indicate the flow of information from the central nervous system ( CNS ) to the periphery . Alpha motor neurons
innervate extrafusal muscle fibers , which are the main force - generating component of a muscle . Their cell bodies are in the ventral horn of the spinal cord
and they are sometimes called ventral horn cells . A single motor neuron may synapse with 150 muscle fibers on average . The motor neuron and all of the muscle
fibers to which it connects is a motor unit .
</item 3>

<item 4>
Motor neuron
A tetanic contraction is caused by constant , very high frequency stimulation - the action potentials come at such a rapid rate that individual twitches are
indistinguishable , and tension rises smoothly eventually reaching a plateau . Contents ( hide ) 1 Development 2 Anatomy and physiology 2.1 Upper motor neurons
2.2 White Matter Tract 2.3 Lower Motor Neurons 2.4 Somatic motor neurons 2.4. 1 Neuromuscular Junction 2.5 Special visceral motor neurons 2.6 General visceral
motor neurons 3 Motor neurons in regenerative medicine 4 See also 5 References 6 Sources Development ( edit ) Motor neurons begin to develop early in embryonic
development , and motor function continues to develop well into childhood . In the neural tube cells are specified to either the rostral - caudal axis or
ventral - dorsal axis . The axons of motor neurons begin to appear in the fourth week of development from the ventral region of the ventral - dorsal axis ( the
basal plate ) . This homeodomain is known as the motor neural progenitor domain ( pMN ) .
</item 4>

<item 5>
Cellular respiration
The pyruvate is not transported into the mitochondrion , but remains in the cytoplasm , where it is converted to waste products that may be removed from the
cell . This serves the purpose of oxidizing the electron carriers so that they can perform glycolysis again and removing the excess pyruvate . Fermentation
oxidizes NADH to NAD+ so it can be re-used in glycolysis . In the absence of oxygen , fermentation prevents the buildup of NADH in the cytoplasm and provides
NAD+ for glycolysis . This waste product varies depending on the organism . In skeletal muscles , the waste product is lactic acid . This type of fermentation
is called lactic acid fermentation . In strenuous exercise , when energy demands exceed energy supply , the respiratory chain can not process all of the
hydrogen atoms joined by NADH . During anaerobic glycolysis , NAD+ regenerates when pairs of hydrogen combine with pyruvate to form lactate . Lactate formation
is catalyzed by lactate dehydrogenase in a reversible reaction .
</item 5>

<item 6>
Motor neuron
There are two types of motor neuron -- upper motor neurons and lower motor neurons . Axons from upper motor neurons synapse onto interneurons in the spinal cord
and occasionally directly onto lower motor neurons . The axons from the lower motor neurons are efferent nerve fibers that carry signals from the spinal cord to
the effectors . Types of lower motor neurons are alpha motor neurons , beta motor neurons , and gamma motor neurons . The term ' motor neuron ' is usually
restricted to the lower motor neurons , the efferent nerves that directly innervate muscles . A single motor neuron may innervate many muscle fibres and a
muscle fibre can undergo many action potentials in the time taken for a single muscle twitch . As a result , if an action potential arrives before a twitch has
completed , the twitches can superimpose on one another , either through summation or a tetanic contraction . In summation , the muscle is stimulated
repetitively such that additional action potentials coming from the somatic nervous system arrive before the end of the twitch . The twitches thus superimpose
on one another , leading to a force greater than that of a single twitch .
</item 6>

<item 7>
Motor neuron
These use both oxidative and glycolytic means to gain energy . In addition to voluntary skeletal muscle contraction , alpha motor neurons also contribute to
muscle tone , the continuous force generated by noncontracting muscle to oppose stretching . When a muscle is stretched , sensory neurons within the muscle
spindle detect the degree of stretch and send a signal to the CNS . The CNS activates alpha motor neurons in the spinal cord , which cause extrafusal muscle
fibers to contract and thereby resist further stretching . This process is also called the stretch reflex . Beta motor neurons innervate intrafusal muscle
fibers of muscle spindles , with collaterals to extrafusal fibres . There are two types of beta motor neurons : Slow Contracting - These innervate extrafusal
fibers . Fast Contracting - These innervate intrafusal fibers . Gamma motor neurons innervate intrafusal muscle fibers found within the muscle spindle . They
regulate the sensitivity of the spindle to muscle stretching . With activation of gamma neurons , intrafusal muscle fibers contract so that only a small stretch
is required to activate spindle sensory neurons and the stretch reflex .
</item 7>

<item 8>
Motor neuron
Branch of NS Position Neurotransmitter Somatic n / a Acetylcholine Parasympathetic Preganglionic Acetylcholine Parasympathetic Ganglionic Acetylcholine
Sympathetic Preganglionic Acetylcholine Sympathetic Ganglionic Norepinephrine * * Except fibers to sweat glands and certain blood vessels Motor neuron
neurotransmitters General visceral motor neurons ( edit ) These motor neurons indirectly innervate cardiac muscle and smooth muscles of the viscera ( the
muscles of the arteries ) : they synapse onto neurons located in ganglia of the autonomic nervous system ( sympathetic and parasympathetic ) , located in the
peripheral nervous system ( PNS ) , which themselves directly innervate visceral muscles ( and also some gland cells ) . In consequence : the motor command of
skeletal and branchial muscles is monosynaptic ( involving only one motor neuron , respectively , somatic and branchial , which synapses onto the muscle ) .
</item 8>

<item 9>
Motor neuron
the command of visceral muscles is disynaptic ( involving two neurons : the general visceral motor neuron located in the CNS , which synapses onto a ganglionic
neuron , located in the PNS , which synapses onto the muscle ) . All vertebrate motor neurons are cholinergic , that is , they release the neurotransmitter
acetylcholine . Parasympathetic ganglionic neurons are also cholinergic , whereas most sympathetic ganglionic neurons are noradrenergic , that is , they release
the neurotransmitter noradrenaline . ( see Table ) Motor neurons in Regenerative medicine ( edit ) Human lower motor neurons can be generated in vitro from
embryonic stem cells and induced pluripotent stem cells . See also ( edit ) Betz cell Central chromatolysis Motor dysfunction Nerve References ( edit ) Jump up
^ Tortora , Gerard ; Derrickson , Bryan ( 2014 ) . Principles of Anatomy & Physiology ( 14th ed . ) . New Jersey : John Wiley & Sons , Inc . pp. 406 , 502 , 541
. ISBN 978 - 1 - 118 - 34500 - 9 .
</item 9>

<item 10>
Motor neuron
These are found in the spinal cord and appear white from the myelin surrounding axons . While these tracts exist for both efferent and afferent nerves ,
efferent nerves will carry signals down the spinal cord toward their effector target . These tracts also serve as the place of origin for lower motor neurons .
There are seven major white matter motor tracts to be found in the spinal cord and are listed below : Lateral Corticospinal Tract Rubrospinal Tract Lateral
Reticulospinal Tract Vestibulospinal tract Medial Reticulospinal Tract Tectospinal Tract Anterior Corticospinal Tract Lower motor neurons ( edit ) Lower motor
neurons are those that originate in the spinal cord and directly or indirectly innervate effector targets . The target of these neurons varies , but in the
somatic nervous system the target will be some sort of muscle fiber . There are three primary categories lower motor neurons , which can be further divided in
sub-categories .
</item 10>
=== END ITEM id=58f1061afbb16c60 ===
