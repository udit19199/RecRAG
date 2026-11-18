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

=== ITEM id=107f4d427c6a1626 ===
Gold sentences and facts:
Sentence 0: The Pakistani National Assembly ( Urdu : قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority .
  Fact 0.0: The Pakistani National Assembly is the lower house of the bicameral Majlis-e-Shura.
  Fact 0.1: The Urdu name of the Pakistani National Assembly is Qaumī Asimbli'e Pākistān.
  Fact 0.2: The Majlis-e-Shura also comprises the President of Pakistan.
  Fact 0.3: The Majlis-e-Shura also comprises the Senate, which is the upper house.
  Fact 0.4: The National Assembly convenes at Parliament House in Islamabad.
  Fact 0.5: The Senate convenes at Parliament House in Islamabad.
  Fact 0.6: The National Assembly is a democratically elected body.
  Fact 0.7: The National Assembly has a total of 332 members.
  Fact 0.8: The members of the National Assembly are referred to as Members of the National Assembly (MNAs).
  Fact 0.9: 272 members of the National Assembly are directly elected.
  Fact 0.10: 70 members of the National Assembly hold reserved seats for women and religious minorities.
  Fact 0.11: A political party must secure 172 seats to obtain a majority.
  Fact 0.12: A political party must secure 172 seats to preserve a majority.

Retrieved items:
<item 1>
National Assembly of Pakistan
Seat distribution ( edit ) In the 14th Assembly , the seats have been allocated among the four provinces , federally - administered area , and capital territory
of Pakistan as follows : Province General Women Total Punjab 148 35 183 Sindh 61 14 75 KPK 35 8 43 Balochistan 14 17 FATA 12 - 12 Federal Capital - See also (
edit ) Speaker of National Assembly Senate of Pakistan Politics of Pakistan Prime Minister of Pakistan Constitution of Pakistan President of Pakistan Supreme
Court of Pakistan Pakistan Bar Council Punjab Bar Council Supreme Court Bar Association of Pakistan List of Pakistan National Assembly Seats List of provincial
governments of Pakistan showing provincial assemblies Notes ( edit ) ^ Jump up to : `` PML - N 's Ayaz Sadiq elected Speaker , Murtaza Abbasi Deputy Speaker in
NA '' . Dawn News . 3 June 2013 . Retrieved 6 June 2013 . Jump up ^ `` PPP 's Khursheed Shah declared leader of opposition in NA '' . Dawn News . 7 June 2013 .
Retrieved 7 June 2013 .
</item 1>

<item 2>
National Assembly of Pakistan
National Assembly of Pakistan - wikipedia National Assembly of Pakistan Jump to : navigation , search For List of MNAs elected in Pakistani general election ,
2013 , see List of MNAs elected in Pakistani general election , 2013 . National Assembly of Pakistan قومی اسمبلئ پاکستان ‬ 14th National Assembly of Pakistan
Type Type Lower house of the Parliament of Pakistan Leadership Speaker Ayaz Sadiq ( PML - N ) Since 9 November 2015 Deputy Speaker Javed Abbasi ( PML - N )
Since 3 June 2013 Leader of House Shahid Khaqan Abbasi ( PML - N ) Since 1 August 2017 Leader of Opposition Khurshid Shah ( PPP ) Since 7 June 2013 Structure
Seats 332 ( 3 vacant ) Political groups Government Coalition ( 209 ) PML -- N ( 188 ) JUI -- F ( 13 ) PML -- F ( 5 ) NPP ( 2 ) NP ( 1 ) Opposition ( 130 ) PPP (
47 ) PTI ( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP (
</item 2>

<item 3>
Pakistan’s National Assembly is the elected lower house, with 332 seats: 272 directly elected by universal adult suffrage and 70 reserved for women and
religious minorities, allocated proportionally among parties. A majority requires 172 seats. Members serve five-year terms; the prime minister, not the
president, dissolves the Assembly. The 13th Assembly, elected in 2008, dissolved on 17 March 2013. The Senate has 104 members, equally representing provinces,
serving stagger
</item 3>

<item 4>
National Assembly of Pakistan
The Public Accounts Committee has a special role to review the report of the Auditor General . Senate , the upper house of the Parliament , has equal
representation from the federating units balancing the provincial inequality in the National Assembly , where the number of members is based on population of
the provinces . The Senate 's role is to promote national cohesion and harmony , and work as a stabilizing factor of the federation . The Senate numbers a total
of 104 members who serve six - year terms which are alternated so that half the senators are up for re-election by the electoral college every three years . The
National Assembly consists of 332 members . The Constitution empowers the President to dissolve the National Assembly , but the Senate is not subject to
dissolution . Only the Parliament can amend the Constitution , by two - thirds majority vote separately in each House . Members of National Assembly ( edit )
See also : List of Pakistan National Assembly Seats Qualifications ( edit ) The Constitution of Pakistan lists a number of requirements for members of the
National Assembly in Article 62 .
</item 4>

<item 5>
National Assembly of Pakistan
( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP ( 3 ) ANP ( 2 ) PML -- Q ( 2 ) AJIP ( 1 ) AML ( 1 ) APML ( 1 ) BNP ( 1 ) PML -- Z ( 1 ) QWP ( 1 ) Ind ( 10 ) Elections Voting
system Mixed member majoritarian ( First past the post for most seats , 60 seats reserved for women and 10 seats reserved for religious minorities by
proportional representation ) Last election 11 May 2013 Meeting place Parliament House , Islamabad Website www.na.gov.pk The Pakistani National Assembly ( Urdu
: قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan
and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad .
</item 5>

<item 6>
National Assembly of Pakistan
The committees have also been empowered to invite or summon before it any member or any other person having a special interest in relation to any matter under
its consideration and may hear expert evidence and hold public hearing . Composition and elections ( edit ) The composition of the National Assembly is
specified in Article 51 of the Constitution of Pakistan . There are a total of 332 seats in the National Assembly . Of these , 272 are filled by direct
elections . In addition , the Pakistani Constitution reserves 10 seats for religious minorities and 60 seats for women , to be filled by proportional
representation among parties with more than 5 % of the vote . As of 2006 , there are 72 women members in the Assembly . Members of the National Assembly are
elected by the people in competitive multi-party elections , to be held at most five years apart on universal adult franchise . To be a member of electoral
college , according to Article 62 of the Constitution , candidates must be citizens of Pakistan and not less than 18 years of age . 2013 election ( edit ) Main
article : Pakistani general election , 2013 Results of the Pakistani general election ,
</item 6>

<item 7>
National Assembly of Pakistan
96 % 27 8 0 0 8 Total 45,388,404 100 % 266 266 60 10 336 Withheld : NA - 103 , NA - 237 ; Postponed : NA - 38 ; Re-poll ordered : NA - 46 ; Terminated : NA - 83
, NA - 254 Source : Detailed Results General Elections - 2013 , Reserve Seats Women , Reserve Seats Minorities Dissolution ( edit ) National Assembly can be
dissolved at the initiative of the Prime Minister . If dissolved , new elections are conducted for the Assembly . Article 58 of the Constitution of Pakistan
deals with the dissolution of the Assembly : 58 . Dissolution of the National Assembly : ( 1 ) The President shall dissolve the National Assembly if so advised
by the Prime Minister ; and the National Assembly shall , unless sooner dissolved , stand dissolved at the expiration of forty - eight hours after the Prime
Minister has so advised .
</item 7>

<item 8>
National Assembly of Pakistan
The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a
total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for
women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority . Members are elected through the first - past - the
- post system under universal adult suffrage , representing electoral districts known as National Assembly constituencies . According to the constitution , the
70 seats reserved for women and religious minorities are allocated to the political parties according to their proportional representation . Each National
Assembly is formed for a five - year term , commencing from the date of the first sitting , after which it is automatically dissolved . Currently the National
Assembly can not be dissolved by the President of Pakistan , it is dissolved by the Prime Minister of Pakistan . Election for 13th National Assembly was held on
18 February 2008 . On March 17 , 2013 13th National Assembly was dissolved on completion of its five - year term under Article 52 of the Constitution .
</item 8>

<item 9>
Pakistan’s 14th National Assembly was the Parliament’s lower house, with 332 seats and three vacant. Speaker Ayaz Sadiq and Deputy Speaker Javed Abbasi were
from PML-N; Shahid Khaqan Abbasi led the House, and PPP’s Khurshid Shah led the opposition. The government coalition held 209 seats, led by PML-N (188);
opposition parties held 130, led by PPP (47). Seats included Punjab 183, S
</item 9>

<item 10>
National Assembly of Pakistan
631 0.47 % 0 9 National Peoples Party 197,829 0.44 % 0 10 Pakistan Muslim League ( Q ) 1,409,905 3.11 % 0 0 11 Awami National Party 453,057 1.00 % 0 0 12 Qaumi
Watan Party ( Sherpao ) 46,574 0.10 % 0 0 13 Pakistan Muslim League ( Z ) 128,510 0.28 % 0 0 14 National Party 61,148 0.13 % 0 0 15 Awami Muslim League Pakistan
93,046 0.20 % 0 0 16 Awami Jamhuri Ittehad Pakistan 71,175 0.16 % 0 0 17 Balochistan National Party 63,979 0.14 % 0 0 18 All Pakistan Muslim League 54,231 0.12
% 0 0 19 Independents 5,880,658 12.96 % 27 8 0 0 8 Total 45,388,
</item 10>
=== END ITEM id=107f4d427c6a1626 ===

=== ITEM id=d891d1057560b2cd ===
Gold sentences and facts:
Sentence 0: Prior to 1940 the term `` jeep '' had been used as U.S. Army slang for new recruits or vehicles , but the World War II `` jeep '' that went into production in 1941 specifically tied the name to this light military 4x4 , arguably making them the oldest four - wheel drive mass - production vehicles now known as SUVs . The Jeep became the primary light 4 - wheel - drive vehicle of the United States Army and the Allies during World War II , as well as the postwar period . The term became common worldwide in the wake of the war . Doug Stewart noted : `` The spartan , cramped , and unstintingly functional jeep became the ubiquitous World War II four - wheeled personification of Yankee ingenuity and cocky , can - do determination . ''
  Fact 0.0: Before 1940, 'jeep' had been used as U.S. Army slang for new recruits or vehicles
  Fact 0.1: The World War II 'jeep' that went into production in 1941 specifically tied the name to a light military 4x4
  Fact 0.2: The World War II jeep arguably made these vehicles the oldest four-wheel drive mass-production vehicles now known as SUVs
  Fact 0.3: The Jeep became the primary light four-wheel-drive vehicle of the United States Army during World War II
  Fact 0.4: The Jeep became the primary light four-wheel-drive vehicle of the Allies during World War II
  Fact 0.5: The Jeep was the primary light four-wheel-drive vehicle of the United States Army and the Allies in the postwar period
  Fact 0.6: The term 'jeep' became common worldwide in the wake of the war
  Fact 0.7: Doug Stewart noted that the spartan, cramped, and unstintingly functional jeep became the ubiquitous World War II four-wheeled personification of Yankee ingenuity and cocky, can-do determination
Sentence 1: At least two Jeep models ( the CJ - 5 and the SJ Wagoneer ) enjoyed extraordinary three - decade production runs of a single body generation . Jeeps have since the war inspired a number of other light utility vehicles , such as the Land Rover . Many Jeep variants serving similar military and civilian roles have since been designed in other nations . In lowercase the term `` jeep '' continues to be used as a generic term for vehicles inspired by the Jeep that are suitable for use on rough terrain .
  Fact 1.0: At least two Jeep models, the CJ-5 and the SJ Wagoneer, enjoyed extraordinary three-decade production runs
  Fact 1.1: The CJ-5 and the SJ Wagoneer each had a production run of a single body generation
  Fact 1.2: Jeeps have inspired a number of other light utility vehicles since the war, such as the Land Rover
  Fact 1.3: Many Jeep variants serving similar military and civilian roles have since been designed in other nations
  Fact 1.4: In lowercase, the term 'jeep' continues to be used as a generic term
  Fact 1.5: The generic term 'jeep' refers to vehicles inspired by the Jeep that are suitable for use on rough terrain
Sentence 2: Jeep 's current product range consists solely of sport utility vehicles and off - road vehicles , but has also included pickup trucks in the past . Some of Jeep 's vehicles , such as the Grand Cherokee and the Wagoneer , reach into the luxury segment . Jeep sold 1.4 million SUVs globally in 2016 , up from 500,000 in 2008 , two - thirds of which in North America , and was Fiat - Chrysler 's best selling brand in the U.S. during the first half of 2017 . In the U.S. alone , over 2400 dealerships hold franchise rights to sell Jeep - branded vehicles , and if Jeep were spun off into a separate company , it is estimated to be worth between $22 and $33.5 billion -- slightly more than all of FCA ( US ) .
  Fact 2.0: Jeep's current product range consists solely of sport utility vehicles and off-road vehicles
  Fact 2.1: Jeep's product range has also included pickup trucks in the past
  Fact 2.2: Some of Jeep's vehicles, such as the Grand Cherokee and the Wagoneer, reach into the luxury segment
  Fact 2.3: Jeep sold 1.4 million SUVs globally in 2016
  Fact 2.4: Jeep sold 500,000 SUVs globally in 2008
  Fact 2.5: Two-thirds of the 1.4 million SUVs Jeep sold globally in 2016 were sold in North America
  Fact 2.6: Jeep was Fiat-Chrysler's best selling brand in the U.S. during the first half of 2017
  Fact 2.7: In the U.S. alone, over 2400 dealerships hold franchise rights to sell Jeep-branded vehicles
  Fact 2.8: If Jeep were spun off into a separate company, it is estimated to be worth between $22 and $33.5 billion
  Fact 2.9: That estimated value is slightly more than all of FCA (US)

Retrieved items:
<item 1>
Jeep
Please consider summarizing the material while citing sources as needed . ( December 2017 ) Main article : List of Jeep vehicles Jeep CJ Main article : Jeep CJ
( Civilian Jeep ) 1982 Jeep Scrambler The CJ ( for `` Civilian Jeep '' ) series were literally the first `` Jeep '' branded vehicles produced for the civilian
public , beginning in 1945 with the CJ - 2A , followed by the CJ - 3A in 1949 and the CJ - 3B in 1953 . These early Jeeps are commonly referred to as ``
flatfenders '' because their front fenders were flat and rectangular across , like on their military forebears , the Willys MB and identical Ford GPW models .
The CJ - 4 exists only as a 1951 prototype , and is the `` missing link '' between the flat - fendered CJ - 2A and CJ - 3B and the round - fendered CJ - 5 first
introduced in 1955 .
</item 1>

<item 2>
Jeep
5 Main articles : Jeep Cherokee ( XJ ) and Jeep Comanche 1984 -- 2001 XJ Cherokee 1984 -- 1990 XJ Wagoneer 1984 -- 1985 Broughwood 1984 -- 1990 Limited 1986 --
1992 Jeep Comanche ( MJ ) Jeep Wrangler 1989 Jeep Wrangler YJ Islander 1997 Jeep Wrangler TJ 2016 Jeep Wrangler JK Unlimited ( MIAS ' 16 ) Jeep Wrangler Main
article : Jeep Wrangler 1987 -- 1995 Jeep Wrangler YJ 1991 -- 1993 Renegade 1988 -- 1995 Wrangler Long -- Venezuela 1995 Wrangler Rio Grande 1997 -- 2006
Wrangler TJ 2002 TJ Se , X , Sport , Sahara models 2003 TJ Rubicon , Rubicon Tomb Raider Edition , Sahara , Sport , X , Se models , Freedom Edition 2004 -- 2006
TJ Long Wheel Base ( LJ ) Unlimited ( 15 '' Longer than a standard TJ ) Rubicon , Sport , X , Se models 2004 -- 2005 Willys Edition ( 2004 -- 1997 made ,
</item 2>

<item 3>
Jeep
2 - door version ; produced by AEV with the Jeep logo . 2007 -- 2018 Wrangler JK 2007 -- 2009 JK Rubicon , Sahara , X 2010 JK Rubicon , Sahara , Mountain ,
Islander , and Sport Editions 2011 70th Anniversary , Call of Duty : Black Ops , Mojave , and Oscar Mike Military ( 200 made ) Editions 2011 JK - 8 Independence
-- a MOPAR JK pick - up conversion kit , mirroring the 1980s CJ - 8 Scrambler pick - up 2013 Rubicon 10th Anniversary Edition 2013 - 2017 Brute Double Cab :
Pickup truck , 4 - door version , produced by American Expedition Vehicles 2014 Willys Wheeler Edition 2017 -- Jeep Wrangler JL Grand Cherokee Main article :
Jeep Grand Cherokee 1st generation Grand Cherokee ZJ . 2008 - 2010 WK Grand Cherokee .
</item 3>

<item 4>
Jeep
Jeep 's current product range consists solely of sport utility vehicles and off - road vehicles , but has also included pickup trucks in the past . Some of Jeep
's vehicles , such as the Grand Cherokee and the Wagoneer , reach into the luxury segment . Jeep sold 1.4 million SUVs globally in 2016 , up from 500,000 in
2008 , two - thirds of which in North America , and was Fiat - Chrysler 's best selling brand in the U.S. during the first half of 2017 . In the U.S. alone ,
over 2400 dealerships hold franchise rights to sell Jeep - branded vehicles , and if Jeep were spun off into a separate company , it is estimated to be worth
between $22 and $33.5 billion -- slightly more than all of FCA ( US ) . Prior to 1940 the term `` jeep '' had been used as U.S.
</item 4>

<item 5>
Jeep
TJL : Compact pickup truck , 2 - door version ; Produced by AAV . JL : Short ( 2 - door ) and long ( 4 - door ) wheelbase SUV ; in production since November
2017 Jeep Grand Cherokee : Mid-size sport utility vehicle Jeep Compass : Compact sport utility vehicle Jeep Cherokee KL : Mid-size sport utility vehicle Future
models Jeep Wagoneer / Grand Wagoneer : Full - Size SUVs expected for 2021 . Jeeps built outside the U.S.A. The Troller T4 Mahindra & Mahindra Limited Indian
Jeep Jeeps have been built and / or assembled around the world by various companies . Argentina -- IKA Jeeps 1956 -- current ; now owned by Chrysler Australia
-- Willys Motors Australia -- 1940s -- 1980s Brazil -- Willys Overland do Brasil , purchased by Ford to become Ford do Brasil -- 1957 -- 1985 built the Jeep
Rural from 1960 -- 1977 , and the Troller T4 is a fiberglass bodied Jeep version built in Brazil . Troller was purchased by Ford do Brasil in 2007 .
</item 5>

<item 6>
Jeep
( prototype ) 1963 Jeep XM - 200 : J200 based concept for developing a low - cost vehicle for third - world countries 1965 Jeep / Renault Model H : A light 4x4
prototype based on the Renault 16 1966 FWD Concept Jeepvair : Similar to the Model H but with a Chevrolet Corvair powertrain 1970 XJ001 1970 XJ002 1971 Jeep
Cowboy : A design study using AMC 's `` compact '' automobile platform 1977 Jeep II 1979 Jeep Jeepster II 1986 Cherokee Targa : A two - door Cherokee
convertible ( later revised as Jeep Freedom show car ) 1987 Comanche Thunderchief : This vehicle was put into production later as the Comanche Eliminator 1989
Jeep Concept 1 : Evolved into the ZJ Grand Cherokee 1989 Jeep Rubicon Wrangler : This vehicle was later put in production 1990 Jeep JJ : Essentially what would
later be called the Icon 1990 Jeep Freedom : A revised Cherokee Targa 1991 Jeep Wagoneer 2000 : A large design concept 1993 Jeep Ecco 1997 Jeep Cherokee
Casablanca : A special edition of Cherokee ,
</item 6>

<item 7>
Jeep
Jeep - wikipedia Jeep Jump to : navigation , search This article is about the U.S. motor manufacturing company . For the original World War II Jeep , see Willys
MB . For other uses , see Jeep ( disambiguation ) . Jeep Type Division Industry Automotive Founded 1943 trademark application 1945 first civilian Jeep launched
Headquarters Toledo , Ohio , U.S. Area served Worldwide Key people Michael Manley ( CEO of Jeep division ) Sergio Marchionne ( CEO of FCA US LLC ) Products
Sport utility vehicles Luxury vehicles Owner Fiat Chrysler Automobiles Parent Willys - Overland ( 1943 - 1953 ) Kaiser - Jeep ( 1953 - 1970 ) American Motors
Corporation ( 1970 - 1986 ) Chrysler ( 1987 - present ) Website jeep.com Jeep is a brand of American automobiles that is a division of FCA US LLC ( formerly
Chrysler Group , LLC ) , a wholly owned subsidiary of Fiat Chrysler Automobiles . Jeep has been a part of Chrysler since 1987 , when Chrysler acquired the Jeep
brand , along with remaining assets , from its previous owner : American Motors Corporation ( AMC ) .
</item 7>

<item 8>
Jeep
windowless hardtop 2014 Jeep Wrangler Level Red 2014 Jeep Cherokee Dakar 2014 Jeep Wrangler MOJO 2015 Jeep Chief 2015 Jeep Wrangler Africa 2015 Jeep Wrangler
Red Rock Responder 2015 Jeep Staff Car : a tribute to Jeep 's military history starting with WWII Current models The Jeep brand currently produces five models ,
but 8 vehicles are under the brand name or use the Jeep logo : Jeep Renegade BU : Subcompact Sport Utility Vehicle Jeep Wrangler JK : Standard wheelbase Compact
Sport utility vehicle , 2 - door version JK Unlimited : Long wheelbase Mid-Size sport utility vehicle , 4 - door version J8 : Mid-Size military sport utility
vehicle ; Produced by AIL , AAV , and AEV . TJL : Compact pickup truck , 2 - door version ; Produced by AAV .
</item 8>

<item 9>
Jeep
Please consider summarizing the material while citing sources as needed . ( April 2018 ) 1944 CJ - 1 prototype 1949 Alcoa Aluminum - bodied Jeepster Coupe (
prototype ) 1949 / 1950 X-98 prototype ; with flat fenders , but a rounded hood and grille like the CJ - 5 , it may have been the first F - head - powered Jeep
1950 CJ - 4 prototype 1950 CJ - 4 M prototype 1950 CJ - 4 MA prototype 1952 CJ Coiler : experimental design for an all independent suspension , with portal -
hub swing - axles and coil - springs 1958 DJ - 3A Pickup : Prototype pickup truck version of the DJ - 3A 1958 Jeep Creep : prototype utility vehicle ; several
versions built for tests , including a Postal rig and an aircraft tug 1959 Jeep J - 100 Malibu and Berkeley : Later developed into the Wagoneer 1960 Jeep Wide -
Trac : Concept for developing a low - cost vehicle for third - world countries 1962 The Brazilian Jeepster ( prototype ) 1963 Jeep XM - 200 : J200 based concept
for developing a
</item 9>

<item 10>
Jeep
With its solid axles and open top , the Wrangler has been called the Jeep model that is as central to the brand 's identity as the rear - engined 911 is to
Porsche . At least two Jeep models ( the CJ - 5 and the SJ Wagoneer ) enjoyed extraordinary three - decade production runs of a single body generation . Jeeps
have since the war inspired a number of other light utility vehicles , such as the Land Rover . Many Jeep variants serving similar military and civilian roles
have since been designed in other nations . In lowercase the term `` jeep '' continues to be used as a generic term for vehicles inspired by the Jeep that are
suitable for use on rough terrain . Contents ( hide ) 1 World War II Jeeps 1.1 Development -- 1 . Bantam Reconnaissance Car 1.2 Development -- 2 . Enter Willys
and Ford 1.3 Full production -- Willys MB and Ford GPW 2 Post-war military Jeeps 2.1 CJ - V35 / U 2.2 M715 3 Jeep etymology 4 Brand , trademarks and image 4.1
Off - road abilities 5 Company history and ownership 5.
</item 10>
=== END ITEM id=d891d1057560b2cd ===

=== ITEM id=738334add98019d7 ===
Gold sentences and facts:
Sentence 0: The Osborne 1 is considered the first true mobile computer by most historians . Adam Osborne founded Osborne Computer and formed the Osborne 1 in 1981 . The Osborne 1 had a five - inch screen , incorporating a modem port , two 5 1 / 4 floppy drives , a big collection of bundled software applications , and a battery pack . The computer company was a failure and did not last for very long . Although it was large and heavy compared to today 's laptops , with a tiny 5 '' CRT monitor , it had a near - revolutionary impact on business , as professionals were able to take their computer and data with them for the first time . This and other `` luggables '' were inspired by what was probably the first portable computer , the Xerox NoteTaker . The Osborne was about the size of a portable sewing machine , and could be carried on commercial aircraft . The Osborne 1 weighed close to 11 kg and was priced at $1795 .
  Fact 0.0: The Osborne 1 is considered the first true mobile computer by most historians.
  Fact 0.1: Adam Osborne founded Osborne Computer.
  Fact 0.2: Adam Osborne formed the Osborne 1 in 1981.
  Fact 0.3: The Osborne 1 had a five-inch screen.
  Fact 0.4: The Osborne 1 incorporated a modem port.
  Fact 0.5: The Osborne 1 had two 5 1/4 floppy drives.
  Fact 0.6: The Osborne 1 came with a big collection of bundled software applications.
  Fact 0.7: The Osborne 1 had a battery pack.
  Fact 0.8: The Osborne Computer company was a failure.
  Fact 0.9: The Osborne Computer company did not last for very long.
  Fact 0.10: Compared to today's laptops, the Osborne 1 was large and heavy.
  Fact 0.11: The Osborne 1 had a tiny 5-inch CRT monitor.
  Fact 0.12: The Osborne 1 had a near-revolutionary impact on business.
  Fact 0.13: Professionals were able to take their computer and data with them for the first time because of the Osborne 1.
  Fact 0.14: Other 'luggables' were inspired by the Osborne 1.
  Fact 0.15: The Osborne 1 was inspired by what was probably the first portable computer, the Xerox NoteTaker.
  Fact 0.16: The Osborne was about the size of a portable sewing machine.
  Fact 0.17: The Osborne could be carried on commercial aircraft.
  Fact 0.18: The Osborne 1 weighed close to 11 kg.
  Fact 0.19: The Osborne 1 was priced at $1795.
Sentence 1: The portable micro computer the `` Portal '' of the French company R2E Micral CCMC officially appeared in September 1980 at the Sicob show in Paris . The Portal was a portable microcomputer designed and marketed by the studies and developments department of the French firm R2E Micral in 1980 at the request of the company CCMC specializing in payroll and accounting . It was based on an Intel 8085 processor , 8 - bit , clocked at 2 MHz . It was equipped with a central 64K byte Ram , a keyboard with 58 alpha numeric keys and 11 numeric keys ( separate blocks ) , a 32 - character screen , a floppy disk : capacity = 140 000 characters , of a thermal printer : speed = 28 characters / second , an asynchronous channel , a synchronous channel , a 220V power supply . Designed for an operating temperature of 15 ° C to 35 ° C , it weighed 12 kg and its dimensions were 45cm x 45cm x 15cm . It provided total mobility . Its operating system was Prologue .
  Fact 1.0: The portable micro computer the 'Portal' of the French company R2E Micral CCMC officially appeared in September 1980 at the Sicob show in Paris.
  Fact 1.1: The Portal was a portable microcomputer designed and marketed by the studies and developments department of the French firm R2E Micral in 1980.
  Fact 1.2: The Portal was designed at the request of the company CCMC, which specializes in payroll and accounting.
  Fact 1.3: The Portal was based on an Intel 8085 processor.
  Fact 1.4: The Portal's Intel 8085 processor was 8-bit and clocked at 2 MHz.
  Fact 1.5: The Portal was equipped with a central 64K byte RAM.
  Fact 1.6: The Portal had a keyboard with 58 alphanumeric keys and 11 numeric keys in separate blocks.
  Fact 1.7: The Portal had a 32-character screen.
  Fact 1.8: The Portal had a floppy disk with a capacity of 140 000 characters.
  Fact 1.9: The Portal had a thermal printer with a speed of 28 characters per second.
  Fact 1.10: The Portal had an asynchronous channel and a synchronous channel.
  Fact 1.11: The Portal required a 220V power supply.
  Fact 1.12: The Portal was designed for an operating temperature of 15 °C to 35 °C.
  Fact 1.13: The Portal weighed 12 kg.
  Fact 1.14: The Portal's dimensions were 45cm x 45cm x 15cm.
  Fact 1.15: The Portal provided total mobility.
  Fact 1.16: The Portal's operating system was Prologue.
Sentence 2: The first laptop in the modern form was the Grid Compass 1101 , designed by Bill Moggridge in 1979 -- 80 , and released in 1983 . Enclosed in a magnesium case , it introduced the now familiar clamshell design , in which the flat display folded shut against the keyboard . The computer could be run from batteries , and was equipped with a 320 × 200 - pixel electroluminescent display and 384 kilobyte bubble memory . It was not IBM - compatible , and its high price ( US $ 8,000 -- 10,000 ) limited it to specialized applications . However , it was used heavily by the U.S. military , and by NASA on the Space Shuttle during the 1980s . The GRiD 's manufacturer subsequently earned significant returns on its patent rights as its innovations became commonplace . GRiD Systems Corp. was later bought by the Tandy ( now RadioShack ) Corporation .
  Fact 2.0: The first laptop in the modern form was the Grid Compass 1101.
  Fact 2.1: The Grid Compass 1101 was designed by Bill Moggridge in 1979-80.
  Fact 2.2: The Grid Compass 1101 was released in 1983.
  Fact 2.3: The Grid Compass 1101 was enclosed in a magnesium case.
  Fact 2.4: The Grid Compass 1101 introduced the now familiar clamshell design, in which the flat display folded shut against the keyboard.
  Fact 2.5: The Grid Compass 1101 could be run from batteries.
  Fact 2.6: The Grid Compass 1101 had a 320 × 200-pixel electroluminescent display.
  Fact 2.7: The Grid Compass 1101 had 384 kilobyte bubble memory.
  Fact 2.8: The Grid Compass 1101 was not IBM-compatible.
  Fact 2.9: The Grid Compass 1101 had a high price of US $8,000-10,000, which limited it to specialized applications.
  Fact 2.10: The Grid Compass 1101 was used heavily by the U.S. military.
  Fact 2.11: The Grid Compass 1101 was used by NASA on the Space Shuttle during the 1980s.
  Fact 2.12: The GRiD manufacturer subsequently earned significant returns on its patent rights as its innovations became commonplace.
  Fact 2.13: GRiD Systems Corp. was later bought by the Tandy (now RadioShack) Corporation.

Retrieved items:
<item 1>
History of laptops
IBM PC Convertible ( edit ) Also among the first commercial IBM - compatible laptops was the IBM PC Convertible , introduced in 1986 . It had a CGA - compatible
LCD and 2 floppy drives . It weighed 13 pounds ( 5.9 kg ) . Toshiba T1100 , T1000 , and T1200 ( edit ) Toshiba launched the Toshiba T1100 in 1985 , and has
subsequently described it as `` the world 's first mass - market laptop computer '' . It did not have a hard drive , and ran entirely from floppy disks . The
CPU was a 4.77 MHz Intel 80C88 , a variation of the popular Intel 8088 , and the display was a monochrome , text - only 640x200 LCD . It was followed in 1987 by
the T1000 and T1200 . Although limited floppy - based DOS machines , with the operating system stored in ROM , the Toshiba models were small and light enough to
be carried in a backpack , and could be run from lead - acid batteries .
</item 1>

<item 2>
History of laptops
It was not IBM - compatible , and its high price ( US $ 8,000 -- 10,000 ) limited it to specialized applications . However , it was used heavily by the U.S.
military , and by NASA on the Space Shuttle during the 1980s . The GRiD 's manufacturer subsequently earned significant returns on its patent rights as its
innovations became commonplace . GRiD Systems Corp. was later bought by the Tandy ( now RadioShack ) Corporation . Dulmont Magnum / Kookaburra ( edit ) Main
article : Dulmont Magnum Another contender for the first true laptop was the Dulmont Magnum , designed Barry Wilkinson and Terry Crews Engineering Manager at
Dulmison in 1982 and released in Australia in 1983 It included an 80 × 8 display in a lid that closed against the keyboard . It was based on the MS - DOS
operating system and applications stored in ROM ( A :) and also supported removable modules in expansion slots ( B : and C :) that could be custom programmed
EPROM or standard word processing and spreadsheet applications .
</item 2>

<item 3>
History of laptops
Another significant machine announced in 1981 , although first sold widely in 1983 , was the Epson HX - 20 . A simple handheld computer , it featured a full -
transit 68 - key keyboard , rechargeable nickel - cadmium batteries , a small ( 120 × 32 - pixel ) dot - matrix LCD with 4 lines of text , 20 characters per
line text mode , a 24 column dot matrix printer , a Microsoft BASIC interpreter , and 16 KB of RAM ( expandable to 32 KB ) . Through at least the late 1980s
Epson continued to release laptops such as the L3s . GRiD Compass ( edit ) Main article : Grid Compass The first laptop in the modern form was the Grid Compass
1101 , designed by Bill Moggridge in 1979 -- 80 , and released in 1983 . Enclosed in a magnesium case , it introduced the now familiar clamshell design , in
which the flat display folded shut against the keyboard . The computer could be run from batteries , and was equipped with a 320 × 200 - pixel
electroluminescent display and 384 kilobyte bubble memory .
</item 3>

<item 4>
History of laptops
Both had LCDs , and could connect to optional external printers . The Dulmont Magnum , launched internationally in 1984 , was an Australian portable similar in
layout to the Gavilan , which used the Intel 80186 processor . Kyotronic 85 ( edit ) The year 1983 also saw the launch of what was probably the biggest -
selling early laptop , the Kyocera Kyotronic 85 . Owing much to the design of the previous Epson HX - 20 , and although at first a slow seller in Japan , it was
quickly licensed by Tandy Corporation , Olivetti , and NEC , who recognised its potential and marketed it respectively as the TRS - 80 Model 100 line ( or Tandy
100 ) , Olivetti M - 10 , and NEC PC - 8201 . The machines ran on standard AA batteries . The Tandy 's built - in programs , including a BASIC interpreter , a
text editor , and a terminal program , were supplied by Microsoft , and was written in part by Bill Gates himself .
</item 4>

<item 5>
History of laptops
Commodore SX - 64 ( edit ) The Commodore SX - 64 , also known as the Executive 64 , or VIC - 64 in Europe , was a portable , briefcase or suitcase - size ``
luggable '' version of the popular Commodore 64 home computer , and was the first full - color portable computer . The SX - 64 featured a built - in five - inch
composite monitor and a built - in 1541 floppy drive . It weighed 20 pounds . The machine was carried by its sturdy handle , which doubled as an adjustable
stand . It was announced in January 1983 and released a year later , at $995 USD . Kaypro 2000 ( edit ) Possibly the first commercial IBM - compatible laptop
was the Kaypro 2000 , introduced in 1985 . With its brushed aluminum clamshell case , it was remarkably similar in design to modern laptops . It featured a 25
line by 80 character LCD , a detachable keyboard , and a pop - up 90 mm ( 3.5 - inch ) floppy drive .
</item 5>

<item 6>
History of laptops
Retrieved 11 August 2015 . Jump up ^ See TRS - 80 Model 100 / 102 at old-computers.com Jump up ^ `` Commodore SX - 64 Portable '' . Retrieved 11 August 2015 .
Jump up ^ `` Commodore SX - 64 portable computer '' . Retrieved 11 August 2015 . Jump up ^ `` World 's First Laptop PC '' . Toshiba Science Museum . Retrieved
21 May 2017 . Jump up ^ `` Business Desktops / Calcs '' . HP Computer Museum . Retrieved 7 April 2014 . Jump up ^ Lewis , Peter H. ( 23 October 1988 ) . `` THE
EXECUTIVE COMPUTER ; Compaq Finally Makes a Laptop '' . The New York Times . Jump up ^ Somerson , Paul ( July 1986 ) . `` Is It On Yet ? '' . PC . pp. 122 --
123 . Retrieved 9 January 2015 . Jump up ^ `` IBM Power Systems archived white papers '' . Retrieved 11 August 2015 .
</item 6>

<item 7>
History of laptops
Cambridge Z88 ( edit ) Another notable computer was the Cambridge Z88 , designed by Clive Sinclair , introduced in 1988 . About the size of an A4 sheet of paper
as well , it ran on standard batteries , and contained basic spreadsheet , word processing , and communications programs . It anticipated the future
miniaturization of the portable computer , and as a ROM - based machine with a small display , can -- like the TRS - 80 Model 100 -- also be seen as a
forerunner of the personal digital assistant . Compaq SLT / 286 ( edit ) By the end of the 1980s , laptop computers were becoming popular among business people
. The COMPAQ SLT / 286 debuted in October 1988 , being the first battery - powered laptop to support an internal hard disk drive and a VGA compatible LCD screen
. It weighed 14 lbs . NEC Ultralite ( edit ) Main article : NEC UltraLite The NEC UltraLite , released in mid-1989 , was perhaps the first `` notebook ''
computer , weighing just 2 kg ( 4.4 lbs ) .
</item 7>

<item 8>
History of laptops
They also introduced the now - standard `` resume '' feature to DOS - based machines : the computer could be paused between sessions without having to be
restarted each time . U.S. Air Force ( edit ) The first laptops successful on a large scale came in large part due to a Request For Proposal ( RFP ) by the U.S.
Air Force in 1987 . This contract would eventually lead to the purchase of over 200,000 laptops . Competition to supply this contract was fiercely contested and
the major PC companies of the time ; IBM Corporation , To shins , Compaq , NEC , and Zenith Data Systems ( ZDS ) , rushed to develop laptops in an attempt to
win this deal . ZDS , which had earlier won a landmark deal with the IRS for its Z - 171 , was awarded this contract for its SupersPort series . The SupersPort
series was originally launched with an Intel 8086 processor , dual floppy disk drives , a backlit , blue and white STN LCD screen , and a NiCd battery pack .
Later models featured an Intel 80286 processor and a 20 MB hard disk drive .
</item 8>

<item 9>
History of laptops
Later models featured an Intel 80286 processor and a 20 MB hard disk drive . On the strength of this deal , ZDS became the world 's largest laptop supplier in
1987 and 1988 . ZDS partnered with Tottori Sanyo in the design and manufacturing of these laptops . This relationship is notable because it was the first deal
between a major brand and an Asian original equipment manufacturer . Hewlett - packard Vectra portable CS ( edit ) In 1987 , HP released a portable version of
their Vectra CS computer . It had the classic laptop configuration ( keyboard and monitor closes up clam - shell style in order to carry ) , however , it was
very heavy and fairly large . It had a full - size keyboard ( with separate numeric keypad ) and a large amber LCD screen . While it was offered with dual 3.5 -
inch floppy disk drives , the most common configuration was a 20 MB hard drive and a single floppy drive . It was one of the first machines with a 1.44 MB
density 3.5 - inch disk drive .
</item 9>

<item 10>
History of laptops
Before laptop / notebook computers were technically feasible , similar ideas had been proposed , most notably Alan Kay 's Dynabook concept , developed at Xerox
PARC in the early 1970 s . One of the first reasonably portable computers was the Xerox NoteTaker , again developed at Xerox PARC , in 1976 . However , only 10
prototypes were built . Contents ( hide ) 1 Portal R2E CCMC 2 Osborne 1 3 Bondwell 2 4 Compaq Portable 5 Epson HX - 20 6 Grid Compass 7 Dulmont Magnum /
Kookaburra 8 Ampere 9 Tandy Model 100 10 Sharp and Gavilan 11 Kyotronic 85 12 Commodore SX - 64 13 Kaypro 2000 14 IBM PC Convertible 15 Toshiba T1100 , T1000 ,
and T1200 16 US Air Force 17 Hewlett - Packard Vectra Portable CS 18 Cambridge Z88 19 Compaq SLT / 286 20 NEC UltraLite 21 Apple 21.1 Macintosh Portable 21.
</item 10>
=== END ITEM id=738334add98019d7 ===

=== ITEM id=2198037c02a75501 ===
Gold sentences and facts:
Sentence 0: Vaugier appeared in Need for Speed : Carbon as Nikki . She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) .
  Fact 0.0: Vaugier appeared in Need for Speed: Carbon as Nikki.
  Fact 0.1: Nikki is the girlfriend of the antagonist in Need for Speed: Carbon.
  Fact 0.2: Nikki is the love interest of the protagonist in Need for Speed: Carbon.
Sentence 1: Year Title Role Notes 2006 Need for Speed : Carbon Nikki Voice , Mo - Cap , & Likeness 2015 Dying Light Jade Mo - Cap 2015 Hellraid Maine , The Female Kin Voice
  Fact 1.0: The filmography table lists a 2006 role as Nikki in Need for Speed: Carbon, with notes Voice, Mo-Cap, & Likeness.
  Fact 1.1: The filmography table lists a 2015 role as Jade in Dying Light, with notes Mo-Cap.
  Fact 1.2: The filmography table lists a 2015 role as Maine, The Female Kin in Hellraid, with notes Voice.

Retrieved items:
<item 1>
Emmanuelle Vaugier
She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) . Filmography ( edit ) Film ( edit ) Year Title Role Notes 1998
Hysteria Veronica Bloom 1999 Shapeshifter Anika 1999 The Fear : Resurrection Jennifer 2000 My 5 Wives Sarah 2000 The Sculptress Sylvie 2001 Mindstorm Tracy
Wellman 2001 Wishmaster 3 : Beyond the Gates of Hell Elinor Smith Direct to video 2001 Ripper Andrea `` Andy '' Carter 2001 Suddenly Naked Lupe Martinez 2002 40
Days and 40 Nights Susie 2003 Secondhand Lions Jasmine 2003 Water 's Edge Rae Butler 2005 House of the Dead II Dr.
</item 1>

<item 2>
Krysten Ritter
The The Last International Playboy Ozzy 2008 27 Dresses Gina the goth 2008 What Happens in Vegas Kelly 2009 Glock Beretta Short film 2009 Confessions of a
Shopaholic Suze Cleath - Stuart She 's Out of My League Patty How to Make Love to a Woman Lauren 2011 Killing Bono Gloria 2011 Life Happens Kim Also writer and
co-producer 2011 Margaret Shopgirl 2012 BuzzKill Nicole 2012 Vamps Stacy Daimen 2012 Refuge Amy 2014 Listen Up Philip Melanie 2014 Veronica Mars Gia Goodman
2014 Asthma Ruby 2014 Search Party Christy 2014 Big Eyes DeAnn 2017 Hero , The The Hero Lucy Television ( edit ) Year Title Role Notes Whoopi Brynn Episode : ``
The Squatters '' One Life to Live Kay 4 episodes Law & Order Tracy Warren Episode : `` Everybody Loves Raimondo 's '' Tanner on Tanner Saleswoman 2 episodes
2005 Jonny Zero Quinn Episode : `` Pilot '' 2005 -- 2006 Veronica Mars Gia Goodman Recurring role ; 8 episodes 2006 Bedford Diaries ,
</item 2>

<item 3>
Emmanuelle Vaugier
Alexandra `` Nightingale '' Morgan 2005 Saw II Addison Corday 2007 Unearthed Sheriff Annie Flynn 2007 Saw IV Addison Corday 2008 Blonde and Blonder Cat 2008
Bachelor Party 2 : The Last Temptation Eva Direct to video 2008 Far Cry Valerie Cardinal 2009 Dolan 's Cadillac Elizabeth Mirrors 2 Elizabeth Reigns Direct to
video Hysteria Jennifer A Nanny for Christmas Ally Leeds 2011 Where the Road Meets the Sun Lisa 2011 French Immersion Jennifer Yates 2012 Stolen Child Amanda
Direct to video 2013 Absolute Deception Rebecca 2013 Susie 's Hope Donna Smith Lawrence 2014 Teen Lust Shelley 2014 Saul : The Journey to Damascus Mary
Magdalene 2015 My Stepdaughter Jill 2016 Destruction : Los Angeles Margot Taylor Television ( edit ) Year Title Role Notes 1995 A Family Divided Rosalie Frank
TV movie 1995 Highlander : The Series Maria Alcobar Episode 4.10 : `` Chivalry '' Madison Noella D'Angelo Recurring role ;
</item 3>

<item 4>
Emmanuelle Vaugier
She appeared in the film 40 Days and 40 Nights , which starred Josh Hartnett . She had starring roles in Suddenly Naked , in which she plays a Latin pop
sensation ; the TV - movie Mindstorm , a science - fiction thriller ; and Ripper , a psychological thriller about students who mysteriously disappear after
enrolling in a class about serial killers . She stars also in the sequel to Mirrors , which is directed by Victor Garcia . Maxim magazine featured her on the
cover of the February 2006 issue in the United States . Three months later , she landed spot # 31 on their annual Hot 100 list . Games ( edit ) Vaugier appeared
in Need for Speed : Carbon as Nikki . She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) .
</item 4>

<item 5>
The Divergent Series
Cast ( edit ) On March 11 , 2013 , it was announced that the roles of Tori , Christina , and Caleb had gone to Maggie Q , Zoë Kravitz , and Ansel Elgort ,
respectively . Ray Stevenson , Jai Courtney and Aaron Eckhart were announced to be in talks to join the cast on March 15 , 2013 , Stevenson and Courtney joined
the cast as Marcus Eaton and Eric , respectively . That same day , Miles Teller was cast as Peter . The studio 's announcement included a bit of casting news :
Mekhi Phifer will play Dauntless leader Max . Kate Winslet was announced to be in talks on January 24 , 2013 . Later it was confirmed that she would portray
Jeanine Matthews . Talking about playing the negative character for the first time , Winslet said , `` I 'm no idiot . The idea went through my head that I have
never played a baddie before , I was almost kind of surprised . '' Since Winslet joined the filming late , she used that distance from her co-stars to appear
aloof on the first day of her shoot .
</item 5>

<item 6>
Emmanuelle Vaugier
Stine 's The Haunting Hour Abigail Episode 1.06 : `` The Red Dress '' 2011 Killer Mountain Kate TV movie 2012 It 's Christmas , Carol ! Carol Huffman TV movie
2013 The Mentalist Melissa Enfield Episode 5.16 : `` There Will Be Blood '' 2013 Clara 's Deadly Secret Helen TV movie 2013 Hidden Away Alexandra Cole /
Stephanie Bennett TV movie 2015 My Stepdaughter Jill TV movie 2015 Mistresses Niko Recurring role ; 6 episodes 2016 Love In Paradise Heather Twain TV movie 2016
Stranger in the House Jade TV movie 2016 His Double Life Linda TV movie 2016 Love in the Vineyard ( Heart Felt ) Molly TV movie 2017 Rogue Regan Recurring role
; 7 episodes 2017 Washed Away Gabrielle Video Games ( edit ) Year Title Role Notes 2006 Need for Speed : Carbon Nikki Voice , Mo - Cap , & Likeness 2015 Dying
Light Jade Mo - Cap 2015 Hellraid Maine , The Female Kin Voice References ( edit ) Jump up ^ `` Archived copy '' .
</item 6>

<item 7>
Emmanuelle Vaugier
She has an older brother , Jason Vaugier . Career ( edit ) Television ( edit ) Vaugier played the role of young , upstart model Maria Alcobar in a 1996 episode
of Highlander : The Series . In 2004 , she played Lindsey Kellogg on the short - lived Fox Television series North Shore . She has also had guest - star spots
on Veronica Mars , Supernatural , The Outer Limits , Higher Ground , Andromeda , and Charmed . She also appeared in the Emmy Award nominated miniseries The
Beach Boys : An American Family where she portrayed Mike Love 's wife , Suzanne . Vaugier also completed a starring role in MOW Veiled Truth for the Lifetime
Network . She is also known for her role during the first two seasons of One Tree Hill , as Nicki , Jake 's ( Bryan Greenberg ) ex-girlfriend and mother to
their child , Jenny . From 2005 - 2015 , Vaugier guest starred in Twelve episodes of CBS 's hit sitcom Two and a Half Men as Mia , a ballet teacher with whom
Charlie Harper fell in love and almost married .
</item 7>

<item 8>
Emmanuelle Vaugier
10 : `` Chivalry '' Madison Noella D'Angelo Recurring role ; 9 episodes The Halfback of Notre Dame Esmerelda TV movie Home Song Cheerleader TV movie The Limbic
Region Jennifer Lucca - age 21 TV movie 1997 Breaker High Monette Episode 1.11 : `` Chateau L'Feet J'mae '' 1997 Police Academy : The Series Sally Episode 1.03
`` Put Down That Nose '' 1997 Ninja Turtles : The Next Mutation Episode 1.12 : `` Turtles ' Night Out '' 1998 First Wave Esther Episode 1.07 : `` Lungfish ''
1998 Dead Man 's Gun Rose Harris Episode 2.10 : `` The Pinkerton '' 1998 -- 99 The Outer Limits Shal ; Lisa Dobkins Episode 4.08 : `` Rite of Passage '' ;
Episode 5.05 : `` The Other Side '' 1999 Viper Mitzi / Olga Episode 3.11 : `` Best Seller '' 1999 Assault on Death Mountain French Woman TV movie 1999 Seven
Days Princess Lisette D'Arcy Episode 2.
</item 8>

<item 9>
The Divergent Series
`` I wanted to break it and say , ' It 's OK , I 'm really fun . I promise . ' But I thought , just for today , I 'd let them think that I am a complete bitch .
'' On March 25 , 2013 , Ben Lamb was cast as Edward , Ben Lloyd - Hughes as Will , and Christian Madsen as Al . In April 2013 , Ashley Judd and Tony Goldwyn
joined the cast as Natalie and Andrew Prior , Beatrice 's parents . In 2014 , the cast for the second film was announced . On May 12 , 2014 , it was announced
that Octavia Spencer joined the cast as Amity representative Johanna Reyes . Late May 2014 , Suki Waterhouse and Jonny Weston were cast as Marlene and Edgar
respectively . Early June 2014 , Stephanie Leigh Schlund announced that she was cast in the film as a member of the Amity , and Naomi Watts and Daniel Dae Kim
joined the cast as Evelyn Johnson and Jack Kang . On June 9 , 2014 , Rosa Salazar joined the cast as Lynn .
</item 9>

<item 10>
Deadpool 2
`` ' Deadpool 2 ' Stunt Crash Victim ID 'd As First African - American Female Pro Road Racer ; Director & Fox `` Deeply Saddened '' -- Update `` . Deadline.com
. Archived from the original on August 15 , 2017 . Retrieved August 15 , 2017 . ^ Jump up to : Patten , Dominic ; Busch , Anita ( August 15 , 2017 ) . `` '
Deadpool 2 ' To Start Production 48 Hours After Stunt Driver Killed '' . Deadline.com . Archived from the original on August 16 , 2017 . Retrieved August 16 ,
2017 . Jump up ^ Johnson , Scott ( August 16 , 2017 ) . `` ' Deadpool 2 ' Motorcycle Death Leaves Stunt Community Asking Tough Questions '' . Deadline.com .
Archived from the original on August 17 , 2017 . Retrieved August 17 , 2017 . Jump up ^ Reynolds , Ryan ( @ VancityReynolds ) ( August 1 , 2017 ) .
</item 10>
=== END ITEM id=2198037c02a75501 ===

=== ITEM id=cf4017d70befb0be ===
Gold sentences and facts:
Sentence 0: A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three vanishing points .
  Fact 0.0: A vanishing point is a point on the image plane of a perspective drawing.
  Fact 0.1: In a perspective drawing, the two-dimensional perspective projections, or drawings, of mutually parallel lines in three-dimensional space appear to converge at a vanishing point.
  Fact 0.2: When the set of parallel lines is perpendicular to a picture plane, the construction is known as one-point perspective.
  Fact 0.3: In one-point perspective, the vanishing point corresponds to the oculus, or "eye point", from which the image should be viewed for correct perspective geometry.
  Fact 0.4: Traditional linear drawings use objects with one to three sets of parallels.
  Fact 0.5: Traditional linear drawings define one to three vanishing points.

Retrieved items:
<item 1>
Vanishing point
Using perspective pinhole projections , a point on L projected on the image plane will have coordinates defined as , x ′ = f x / z = f d − bz / az y ′ = f y / z
= f c / z This is the parametric representation of the image L ′ of the line L with z as the parameter . When z → − ∞ it stops at the point ( x ′ , y ′ ) = ( −
fb / a , 0 ) on the x ′ axis of the image plane . This is the vanishing point corresponding to all parallel lines with slope − b / a in the plane π . All
vanishing points associated with different lines with different slopes belonging to plane π will lie on the x ′ axis , which in this case is the horizon line .
2 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x , y , f ) , v ≡ ( x , y , f ) be the three
corresponding vanishing points respectively .
</item 1>

<item 2>
Drawing
Once these basic volumes have been assembled into a likeness , then the drawing can be refined into a more accurate and polished form . The lines of the
primitive volumes are removed and replaced by the final likeness . Drawing the underlying construction is a fundamental skill for representational art , and is
taught in many books and schools . Its correct application resolves most uncertainties about smaller details , and makes the final image look consistent . A
more refined art of figure drawing relies upon the artist possessing a deep understanding of anatomy and the human proportions . A trained artist is familiar
with the skeleton structure , joint location , muscle placement , tendon movement , and how the different parts work together during movement . This allows the
artist to render more natural poses that do not appear artificially stiff . The artist is also familiar with how the proportions vary depending on the age of
the subject , particularly when drawing a portrait . Two - point perspective drawing Perspective ( edit ) Linear perspective is a method of portraying objects
on a flat surface so that the dimensions shrink with distance . Each set of parallel , straight edges of any object , whether a building or a table , follows
lines that eventually converge at a vanishing point .
</item 2>

<item 3>
Vanishing point
Vanishing point - wikipedia Vanishing point Jump to : navigation , search For other uses , see Vanishing point ( disambiguation ) . A vanishing point can be
seen at the far end of this railroad . A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective
projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a
picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the
image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three
vanishing points . Part of a series on Graphical projection Planar ( show ) Parallel projection Orthographic projection Multiview projection Axonometric
projection Isometric projection Oblique projection Perspective projection Curvilinear perspective Reverse perspective Views ( show ) 2.
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
Vanishing point
If we know the coordinates of one of these points , say v , and the direction of a straight line on the image plane , which passes through a second point , say
v , we can compute the coordinates of both v and v 3 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x ,
y , f ) , v ≡ ( x , y , f ) be the three corresponding vanishing points respectively . The orthocenter of the triangle with vertices in the three vanishing
points is the intersection of the optical axis and the image plane . Curvilinear and reverse perspective ( edit ) A curvilinear perspective is a drawing with
either 4 or 5 vanishing points . In 5 - point perspective the vanishing points are mapped into a circle with 4 vanishing points at the cardinal headings N , W ,
S , E and one at the circle 's origin . A reverse perspective is a drawing with vanishing points that are placed outside the painting with the illusion that
they are `` in front of '' the painting . Single point perspective projection . Double point perspective projection .
</item 5>

<item 6>
Drawing
Typically this convergence point is somewhere along the horizon , as buildings are built level with the flat surface . When multiple structures are aligned with
each other , such as buildings along a street , the horizontal tops and bottoms of the structures typically converge at a vanishing point . When both the fronts
and sides of a building are drawn , then the parallel lines forming a side converge at a second point along the horizon ( which may be off the drawing paper . )
This is a two - point perspective . Converging the vertical lines to a third point above or below the horizon then produces a three - point perspective . Depth
can also be portrayed by several techniques in addition to the perspective approach above . Objects of similar size should appear ever smaller the further they
are from the viewer . Thus the back wheel of a cart appears slightly smaller than the front wheel . Depth can be portrayed through the use of texture . As the
texture of an object gets further away it becomes more compressed and busy , taking on an entirely different character than if it was close . Depth can also be
portrayed by reducing the contrast in more distant objects , and by making their colors less saturated .
</item 6>

<item 7>
Vanishing point
The horizon line is a theoretical line that represents the eye level of the observer . If the object is below the horizon line , its vanishing lines angle up to
the horizon line . If the object is above , they slope down . All vanishing lines end at the horizon line . Properties of vanishing points ( edit ) 1 .
Projections of two sets of parallel lines lying in some plane π appear to converge , i.e. the vanishing point associated with that pair , on a horizon line , or
vanishing line H formed by the intersection of the image plane with the plane parallel to π and passing through the pinhole . Proof : Consider the ground plane
π , as y = c which is , for the sake of simplicity , orthogonal to the image plane . Also , consider a line L that lies in the plane π , which is defined by the
equation ax + bz = d .
</item 7>

<item 8>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 8>

<item 9>
Vanishing point
say D , will have the same vanishing point or converge at the same vanishing points . Mathematically , let q ≡ ( x , y , f ) be a point lying on the image plane
, where f is the focal length ( of the camera associated with the image ) , and let v ≡ ( x / h , y / h , f / h ) be the unit vector associated with q , where h
= √ x + y + f . If we consider a straight line in space S with the unit vector n ≡ ( n , n , n ) and its vanishing point v , the unit vector associated with v
is equal to n , assuming both are assumed to point towards the image plane . When the image plane is parallel to two world - coordinate axes , lines parallel to
the axis which is cut by this image plane will meet at infinity i.e. at the vanishing point . Lines parallel to the other two axes will not form vanishing
points as they are parallel to the image plane . This is one - point perspective .
</item 9>

<item 10>
Vanishing point
She notes , in terms of projective geometry , the vanishing point is the image of the point at infinity associated with L , as the sightline from O through the
vanishing point is parallel to L . Vanishing line ( edit ) As a vanishing point originates in a line , so a vanishing line originates in a plane α that is not
parallel to the picture π . Given the eye point O , and β the plane parallel to α and lying on O , then the vanishing line of α is β ∩ π . For example , when α
is the ground plane and β is the horizon plane , then the vanishing line of α is the horizon line β ∩ π . Anderson notes , `` Only one particular vanishing line
occurs , often referred to as the `` horizon '' . To put it simply , the vanishing line of some plane , say α , is obtained by the intersection of the image
plane with another plane , say β , parallel to the plane of interest ( α ) , passing through the camera center . For different sets of lines parallel to this
plane α , their respective vanishing points will lie on this vanishing line .
</item 10>
=== END ITEM id=cf4017d70befb0be ===

=== ITEM id=d824442247c980ad ===
Gold sentences and facts:
Sentence 0: The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar in 1901 CE , which started as Samvat 1958 . In Nepal , the new year begins with the first day of the month of Baishakh , which usually falls within the months of April -- May in the Gregorian calendar . The first day of the new year is passionately celebrated in a historical carnival that takes place every year in Bhaktapur , called Bisket Jatra.As before , from 2007 AD Nepal Sambat is recognized as the national calender .
  Fact 0.0: The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar.
  Fact 0.1: The Rana rulers made Vikram Samvat the official Hindu calendar in 1901 CE.
  Fact 0.2: Vikram Samvat started as Samvat 1958.
  Fact 0.3: In Nepal the new year begins on the first day of the month of Baishakh.
  Fact 0.4: Baishakh usually falls within April to May in the Gregorian calendar.
  Fact 0.5: The first day of the new year is passionately celebrated in a historical carnival called Bisket Jatra.
  Fact 0.6: The Bisket Jatra carnival takes place every year in Bhaktapur.
  Fact 0.7: As before, from 2007 AD, Nepal Sambat is recognized as the national calendar.
Sentence 1: Vikram Samvat ( Hindi : विक्रम सम्वत् , Nepali : विक्रम सम्वत् ) ( abbreviated as V.S. ( or VS ) or B.S. ( or BS ) ) ; Listen ( help info ) ) is the historical Hindu calendar of India and Nepal . It uses lunar months and solar sidereal year ( see : Vedic time keeping ) . It is used as the official calendar in Nepal .
  Fact 1.0: Vikram Samvat is written in Hindi as विक्रम सम्वत्.
  Fact 1.1: Vikram Samvat is written in Nepali as विक्रम सम्वत्.
  Fact 1.2: Vikram Samvat is abbreviated as V.S. or VS.
  Fact 1.3: Vikram Samvat is abbreviated as B.S. or BS.
  Fact 1.4: Vikram Samvat is the historical Hindu calendar of India and Nepal.
  Fact 1.5: Vikram Samvat uses lunar months.
  Fact 1.6: Vikram Samvat uses the solar sidereal year.
  Fact 1.7: Vikram Samvat is used as the official calendar in Nepal.

Retrieved items:
<item 1>
Vikram Samvat
BRILL . p. 103 . ISBN 90 - 04 - 18566 - 6 . Jump up ^ Falk and Bennett ( 2009 ) , pp. 197 - 215 . Jump up ^ `` Gujarat CM to exchange Diwali - New Year
greetings with people '' . 19 October 2014 . Retrieved 24 October 2014 . Jump up ^ Pancanga ( version 3.14 ) Jump up ^ On a Jain Statue in the Horniman Museum ,
F. Kielhorn , Journal of the Royal Asiatic Society of Great Britain and Ireland , 1898 , p. 101 - 102 Further reading ( edit ) Wikisource has the text of the
1911 Encyclopædia Britannica article Vikramaditya . Harry Falk and Chris Bennett ( 2009 ) . `` Macedonian Intercalary Months and the Era of Azes . '' Acta
Orientalia 70 , pp. 197 -- 215 . `` The Dynastic Art of the Kushan '' , John Rosenfield . `` Samvat '' . New International Encyclopedia . 1905 . ( hide )
Calendars Systems
</item 1>

<item 2>
Vikram Samvat
Converting Vikram Samvat to CE ( edit ) While an approximate conversion can be made by subtracting 56 or 57 , a precise computation requires identification of
the version of Vikam Samvat that was used : Year expired ( default ) or current Kartikadi ( older or western India ) or Chaitradi ( modern Northern India ) .
Kartikadi starts 7 months later in Kartika . Until 12th century Kartikadi year was common even in North India . Months Amanta ( south ) or Purminata ( North ) .
Amanta Chaitra Krishna Paksha is Purminata Vaishakh Krishna Paksha . For the Shukla Paksh , the month is the same . Software named Pancanga is available to make
the calculations . Thus the date Vikram 1208 ( Kartikadi , expired ) , Vaishakh Krishna 5 ( Purnimanta ) , Thursday , is equivalent to Vikram 1209 ( Chaitradi ,
expired ) , Chaitra Krishna 5 ( Amanta ) , and to Thursday March 27 , 1152 .
</item 2>

<item 3>
Vikram Samvat
In India , the reformulated Saka Calendar is officially used ( although not for computing the dates of the traditional festivals ) , although in the Hindi
version of the Preamble of the Constitution of India , the date of adoption of the constitution , 26 November 1949 , is presented in Vikram Samvat ( Margsheersh
Shukla Saptami Samvat 2006 ) . There have been calls for the Vikram Samvat to replace Saka as India 's official calendar . Contents ( hide ) 1 Divisions of a
year 1.1 Lunar metrics 2 History 2.1 Vikramaditya legends 2.2 Historicity of the legends 3 Culture 4 Converting Vikram Samvat to CE 5 See also 6 References 7
Further reading Divisions of a year ( edit ) The classical Vikram Samvat uses lunar months and solar sidereal years . Because 12 months do not match a sidereal
year exactly , correctional months ( adhika māsa ) are added , or occasionally subtracted ( kshaya masa ) .
</item 3>

<item 4>
Vikram Samvat
New International Encyclopedia . 1905 . ( hide ) Calendars Systems Lunar Lunisolar Solar In wide use Astronomical Bengali Chinese Ethiopian Gregorian Hebrew
Hindu Iranian Islamic ISO Unix time In more limited use Akan Armenian Assyrian Bahá'í ( Badí ' ) Balinese pawukon Balinese saka Berber Buddhist Burmese Chinese
Coptic Gaelic Germanic Heathen Georgian Hebrew Hindu or Indian Vikram Samvat Saka Igbo Iranian Jalali ( medieval ) Hijri ( modern ) Zoroastrian Islamic Fasli
Tabular Jain Japanese Javanese Korean Juche Kurdish Lithuanian Malayalam Mongolian Melanau Nanakshahi Nepal Sambat Nisg̱a'a Oromo Romanian Somali Sesotho Slavic
Slavic Native Faith Tamil Thai lunar solar Tibetan Vietnamese Xhosa Yoruba Types Runic Mesoamerican Long Count Calendar round Christian variants Julian Revised
Liturgical year Eastern Orthodox Saints Historical Attic Aztec Tonalpohualli Xiuhpohualli Babylonian Bulgar Byzantine Celtic Cham Culāsakaraj Egyptian
Florentine French Republican Germanic Greek Hindu Inca Macedonian Maya Haab ' Tzolk'in Muisca
</item 4>

<item 5>
Vikram Samvat
It occurs in mid-April each year , and coincides with the traditional new year in Assam , Bengal , Burma , Cambodia , Kerala , Kashmir , Manipur , Odisha ,
Punjab , Sri Lanka , Tamil Nadu and Thailand . In addition to Nepal , the Vikram Samvat calendar is also recognized in North and East India , and in Gujarat
among Hindus . In Buddhist communities , the month of Baishakh is associated with Vesak or Buddha 's Birthday . It commemorates the birth , Enlightenment and
passing of Gautama Buddha on the first full moon day in May , except in a leap year when the festival is held in June . Although this festival is not held on
the same day as Pahela Baishakh , the holidays typically fall in the same month ( Baishakh ) of the Bengali , Hindu , and Theravada Buddhist calendars , and are
related historically through the spread of Hinduism and Buddhism in South Asia . In Gujarat , the second day of Diwali is celebrated as the first day of the
Vikram Samvat calendar which is the first day of the month Kartik .
</item 5>

<item 6>
Vikram Samvat
See also ( edit ) Bikrami calendar Hindu calendars Vira Nirvana Samvat References ( edit ) Jump up ^ Richard Salomon ( 1398 ) . Indian Epigraphy : A Guide to
the Study of Inscriptions in Sanskrit , Prakrit , and the other Indo - Aryan Languages . Oxford University Press . pp. 181 -- 183 . ISBN 978 - 0 - 19 - 535666 -
3 . ^ Jump up to : Ashvini Agrawal ( 1989 ) . Rise and Fall of the Imperial Guptas . Motilal Banarsidass Publ . pp. 174 -- 175 . ISBN 978 - 81 - 208 - 0592 - 7
. Jump up ^ Encyclopedia of New Year 's Holidays Worldwide , William D. Crump , Publisher McFarland , 2014 p. 38 Jump up ^ `` Vikram Samvat should be declared
national calendar '' . The Free Press Journal . 15 February 2012 . Archived from the original on 26 April 2014 . Retrieved 28 March 2012 .
</item 6>

<item 7>
Vikram Samvat
According to Rudolf Hoernlé , the king responsible for this change was Yashodharman : Hoernlé also believed that he conquered Kashmir , and is same person as
the `` Harsha Vikramaditya '' mentioned in Kalhana 's Rajatarangini . Earlier , some scholars believed that the Vikrama Samavat corresponded to the Azes era of
the Indo - Scythian ( Śaka ) king King Azes . However , this was disputed by Robert Bracey following discovery of an inscription of Vijayamitra , which is dated
in two eras . The theory seems to be now thoroughly discredited by Falk and Bennett , who place the inception of the Azes era in 47 -- 46 BCE . Culture ( edit )
The traditional New Year of Vikram Samvat is one of the many festivals of Nepal , marked by parties , family gatherings , the exchange of good wishes , and
participation in rituals to ensure good fortune in the coming year .
</item 7>

<item 8>
Vikram Samvat
Republican Germanic Greek Hindu Inca Macedonian Maya Haab ' Tzolk'in Muisca Pentecontad Pisan Rapa Nui Roman calendar Rumi Soviet Swedish Turkmen By specialty
Holocene ( anthropological ) Proleptic Gregorian / Proleptic Julian ( historiographical ) Darian ( Martian ) Dreamspell ( New Age ) Discordian / Pataphysical (
surreal ) Proposals Calendar reform Hanke -- Henry Permanent International Fixed Pax Positivist Symmetry454 Tranquility World New Earth Time Fictional Discworld
Greyhawk Middle - earth Stardate Star Wars ( Galactic Standard Calendar ) Displays and applications Electronic Perpetual Wall Year naming and numbering
Terminology Era Epoch Regnal name Regnal year Year zero Systems Ab urbe condita Anno Domini / Common Era Anno Mundi Assyrian Before Present Chinese Imperial
Chinese Minguo Human Era Japanese Korean Seleucid Spanish Yugas Satya Treta Dvapara Kali Vietnamese List of calendars Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Vikram_Samvat&oldid=822253746 '' Categories : Specific calendars Nepali calendar Hindu calendar Memorials to
Vikramaditya Calendar eras
</item 8>

<item 9>
Vikram Samvat
The Shukla Paksha in both systems coincides , most festivals occur in the Shukla Paksha . The era is named after King Vikramaditya of India . The lunisolar
Vikram Samvat calendar is 56.7 years ahead ( in count ) of the solar Gregorian calendar . For example , the year 2074 VS began in 2017 CE and will end in 2018
CE . The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar in 1901 CE , which started as Samvat 1958 . In Nepal , the new year begins with the
first day of the month of Baishakh , which usually falls within the months of April -- May in the Gregorian calendar . The first day of the new year is
passionately celebrated in a historical carnival that takes place every year in Bhaktapur , called Bisket Jatra.As before , from 2007 AD Nepal Sambat is
recognized as the national calender .
</item 9>

<item 10>
Vikram Samvat
Vikram Samvat - Wikipedia Vikram Samvat Jump to : navigation , search It has been requested that the title of this article be changed to Bikram Sambat . Please
see the relevant discussion on the discussion page . Do not move the page until the discussion has reached consensus for the change and is closed . It has been
suggested that Bikrami calendar be merged into this article . ( Discuss ) Proposed since October 2017 . Vikram Samvat ( Hindi : विक्रम सम्वत् , Nepali : विक्रम
सम्वत् ) ( abbreviated as V.S. ( or VS ) or B.S. ( or BS ) ) ; Listen ( help info ) ) is the historical Hindu calendar of India and Nepal . It uses lunar months
and solar sidereal year ( see : Vedic time keeping ) . It is used as the official calendar in Nepal . The Vikram Samvat has two alternative systems . It started
in 56 BCE in southern ( purnimanta ) and 57 -- 56 BCE in northern ( amanta ) systems of Hindu calendar .
</item 10>
=== END ITEM id=d824442247c980ad ===
