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

=== ITEM id=13540d1ed48e43f7 ===
Gold sentences and facts:
Sentence 0: City Image Country Population Year Mexico City Mexico 8,918,653 2015 New York City United States 8,550,405 2015 Los Angeles United States 3,971,883 2015 Toronto Canada 2,826,498 2015 Chicago United States 2,720,546 2015 6 Houston United States 2,296,224 2015 7 Havana Cuba 2,117,625 2015 8 Montreal Canada 1,753,034 2015 9 Ecatepec de Morelos Mexico 1,677,678 2015 10 Philadelphia United States 1,567,442 2015 11 Phoenix United States 1,563,025 2015 12 San Antonio United States 1,469,845 2015 13 Guadalajara Mexico 1,460,148 2015 14 Puebla Mexico 1,437,939 2015 15 San Diego United States 1,394,928 2015 16 Juárez Mexico 1,382,753 2015 17 León Mexico 1,349,224 2015 18 Dallas United States 1,300,092 2015 19 Tijuana Mexico 1,298,475 2015 20 Calgary Canada 1,230,915 2015 21 Tegucigalpa Honduras 1,190,230 2015 22 Zapopan Mexico 1,179,681 2015 23 Monterrey Mexico 1,109,171 2015 24 Managua Nicaragua 1,048,134 2015 25 Nezahualcóyotl Mexico 1,039,867 2015 26 San Jose United States 1,026,908 2015 27 Santo Domingo Dominican Republic 1,007,997 2015 28 Guatemala City Guatemala 994,078 2015 29 Port - au - Prince Haiti 987,310 2015 30 Naucalpan Mexico 970,012 2015 31 Ottawa Canada 956,710 2015 32 Austin United States 931,830 2015 33 Edmonton Canada 899,447 2016 34 Mérida Mexico 892,363 2015 35 Querétaro Mexico 878,931 2015 36 Toluca Mexico 873,536 2015 37 Jacksonville United States 868,031 2015 38 Chihuahua Mexico 867,736 2015 39 San Francisco United States 864,816 2015 40 Indianapolis United States 853,173 2015 41 Columbus United States 850,106 2015 42 Fort Worth United States 833,319 2015 43 Charlotte United States 827,097 2015 44 Hermosillo Mexico 819,999 2015 45 Saltillo Mexico 791,074 2015 46 Aguascalientes Mexico 785,942 2015 47 Mississauga Canada 761,300 2015 48 San Luis Potosí Mexico 759,854 2015 49 Veracruz Mexico 752,171 2015 50 San Pedro Sula Honduras 742,118 2015 51 Santiago de los Caballeros Dominican Republic 723,162 2015 52 Culiacán Mexico 720,798 2015 53 Winnipeg Canada 718,400 2015 54 Mexicali Mexico 717,628 2015 55 Cancún Mexico 711,682 2015 56 Acapulco Mexico 710,261 2015 57 Tlalnepantla Mexico 700,734 2015 58 Seattle United States 684,451 2015 59 Guadalupe Mexico 682,880 2015 60 Denver United States 682,545 2015 61 El Paso United States 681,124 2015 62 Chimalhuacán Mexico 679,811 2015 63 Detroit United States 677,116 2015 64 Washington , D.C. United States 672,228 2015 65 Kingston Jamaica 669,627 2015 66 Boston United States 667,137 2015 67 Tlaquepaque Mexico 664,193 2015 68 Memphis United States 655,770 2015 69 Nashville United States 654,610 2015 70 Torreón Mexico 653,150 2015 71 Vancouver Canada 648,608 2015 72 Portland United States 632,309 2015 73 Reynosa Mexico 631,611 2015 74 Oklahoma City United States 631,346 2015 75 Las Vegas United States 623,747 2015 76 Baltimore United States 621,849 2015 77 Brampton Canada 620,700 2016 78 Louisville United States 615,366 2015 79 Morelia Mexico 608,190 2015 80 Milwaukee United States 600,155 2015 81 Tuxtla Gutiérrez Mexico 598,710 2015 82 Apodaca Mexico 597,207 2015 83 Durango Mexico 590,893 2015 84 Villa Nueva Guatemala 564,686 2015 85 Albuquerque United States 559,121 2015 86 Hamilton Canada 556,359 2015 87 Quebec City Canada 540,994 2015 88 Tonalá Mexico 536,111 2015 89 Tucson United States 531,641 2015 90 Cuautitlán Izcalli Mexico 531,041 2015 91 Surrey Canada 526,004 2015 92 Ciudad López Mateos Mexico 523,296 2015 93 Tultitlán Mexico 520,557 2015 94 Fresno United States 520,052 2015 95 Carrefour Haiti 501,768 2015
  Fact 0.0: Mexico City is a city in Mexico with a population of 8,918,653 in 2015.
  Fact 0.1: New York City is a city in the United States with a population of 8,550,405 in 2015.
  Fact 0.2: Los Angeles is a city in the United States with a population of 3,971,883 in 2015.
  Fact 0.3: Toronto is a city in Canada with a population of 2,826,498 in 2015.
  Fact 0.4: Chicago is a city in the United States with a population of 2,720,546 in 2015.
  Fact 0.5: Houston is a city in the United States, ranked 6th, with a population of 2,296,224 in 2015.
  Fact 0.6: Havana is a city in Cuba, ranked 7th, with a population of 2,117,625 in 2015.
  Fact 0.7: Montreal is a city in Canada, ranked 8th, with a population of 1,753,034 in 2015.
  Fact 0.8: Ecatepec de Morelos is a city in Mexico, ranked 9th, with a population of 1,677,678 in 2015.
  Fact 0.9: Philadelphia is a city in the United States, ranked 10th, with a population of 1,567,442 in 2015.
  Fact 0.10: Phoenix is a city in the United States, ranked 11th, with a population of 1,563,025 in 2015.
  Fact 0.11: San Antonio is a city in the United States, ranked 12th, with a population of 1,469,845 in 2015.
  Fact 0.12: Guadalajara is a city in Mexico, ranked 13th, with a population of 1,460,148 in 2015.
  Fact 0.13: Puebla is a city in Mexico, ranked 14th, with a population of 1,437,939 in 2015.
  Fact 0.14: San Diego is a city in the United States, ranked 15th, with a population of 1,394,928 in 2015.
  Fact 0.15: Juárez is a city in Mexico, ranked 16th, with a population of 1,382,753 in 2015.
  Fact 0.16: León is a city in Mexico, ranked 17th, with a population of 1,349,224 in 2015.
  Fact 0.17: Dallas is a city in the United States, ranked 18th, with a population of 1,300,092 in 2015.
  Fact 0.18: Tijuana is a city in Mexico, ranked 19th, with a population of 1,298,475 in 2015.
  Fact 0.19: Calgary is a city in Canada, ranked 20th, with a population of 1,230,915 in 2015.
  Fact 0.20: Tegucigalpa is a city in Honduras, ranked 21st, with a population of 1,190,230 in 2015.
  Fact 0.21: Zapopan is a city in Mexico, ranked 22nd, with a population of 1,179,681 in 2015.
  Fact 0.22: Monterrey is a city in Mexico, ranked 23rd, with a population of 1,109,171 in 2015.
  Fact 0.23: Managua is a city in Nicaragua, ranked 24th, with a population of 1,048,134 in 2015.
  Fact 0.24: Nezahualcóyotl is a city in Mexico, ranked 25th, with a population of 1,039,867 in 2015.
  Fact 0.25: San Jose is a city in the United States, ranked 26th, with a population of 1,026,908 in 2015.
  Fact 0.26: Santo Domingo is a city in the Dominican Republic, ranked 27th, with a population of 1,007,997 in 2015.
  Fact 0.27: Guatemala City is a city in Guatemala, ranked 28th, with a population of 994,078 in 2015.
  Fact 0.28: Port-au-Prince is a city in Haiti, ranked 29th, with a population of 987,310 in 2015.
  Fact 0.29: Naucalpan is a city in Mexico, ranked 30th, with a population of 970,012 in 2015.
  Fact 0.30: Ottawa is a city in Canada, ranked 31st, with a population of 956,710 in 2015.
  Fact 0.31: Austin is a city in the United States, ranked 32nd, with a population of 931,830 in 2015.
  Fact 0.32: Edmonton is a city in Canada, ranked 33rd, with a population of 899,447 in 2016.
  Fact 0.33: Mérida is a city in Mexico, ranked 34th, with a population of 892,363 in 2015.
  Fact 0.34: Querétaro is a city in Mexico, ranked 35th, with a population of 878,931 in 2015.
  Fact 0.35: Toluca is a city in Mexico, ranked 36th, with a population of 873,536 in 2015.
  Fact 0.36: Jacksonville is a city in the United States, ranked 37th, with a population of 868,031 in 2015.
  Fact 0.37: Chihuahua is a city in Mexico, ranked 38th, with a population of 867,736 in 2015.
  Fact 0.38: San Francisco is a city in the United States, ranked 39th, with a population of 864,816 in 2015.
  Fact 0.39: Indianapolis is a city in the United States, ranked 40th, with a population of 853,173 in 2015.
  Fact 0.40: Columbus is a city in the United States, ranked 41st, with a population of 850,106 in 2015.
  Fact 0.41: Fort Worth is a city in the United States, ranked 42nd, with a population of 833,319 in 2015.
  Fact 0.42: Charlotte is a city in the United States, ranked 43rd, with a population of 827,097 in 2015.
  Fact 0.43: Hermosillo is a city in Mexico, ranked 44th, with a population of 819,999 in 2015.
  Fact 0.44: Saltillo is a city in Mexico, ranked 45th, with a population of 791,074 in 2015.
  Fact 0.45: Aguascalientes is a city in Mexico, ranked 46th, with a population of 785,942 in 2015.
  Fact 0.46: Mississauga is a city in Canada, ranked 47th, with a population of 761,300 in 2015.
  Fact 0.47: San Luis Potosí is a city in Mexico, ranked 48th, with a population of 759,854 in 2015.
  Fact 0.48: Veracruz is a city in Mexico, ranked 49th, with a population of 752,171 in 2015.
  Fact 0.49: San Pedro Sula is a city in Honduras, ranked 50th, with a population of 742,118 in 2015.
  Fact 0.50: Santiago de los Caballeros is a city in the Dominican Republic, ranked 51st, with a population of 723,162 in 2015.
  Fact 0.51: Culiacán is a city in Mexico, ranked 52nd, with a population of 720,798 in 2015.
  Fact 0.52: Winnipeg is a city in Canada, ranked 53rd, with a population of 718,400 in 2015.
  Fact 0.53: Mexicali is a city in Mexico, ranked 54th, with a population of 717,628 in 2015.
  Fact 0.54: Cancún is a city in Mexico, ranked 55th, with a population of 711,682 in 2015.
  Fact 0.55: Acapulco is a city in Mexico, ranked 56th, with a population of 710,261 in 2015.
  Fact 0.56: Tlalnepantla is a city in Mexico, ranked 57th, with a population of 700,734 in 2015.
  Fact 0.57: Seattle is a city in the United States, ranked 58th, with a population of 684,451 in 2015.
  Fact 0.58: Guadalupe is a city in Mexico, ranked 59th, with a population of 682,880 in 2015.
  Fact 0.59: Denver is a city in the United States, ranked 60th, with a population of 682,545 in 2015.
  Fact 0.60: El Paso is a city in the United States, ranked 61st, with a population of 681,124 in 2015.
  Fact 0.61: Chimalhuacán is a city in Mexico, ranked 62nd, with a population of 679,811 in 2015.
  Fact 0.62: Detroit is a city in the United States, ranked 63rd, with a population of 677,116 in 2015.
  Fact 0.63: Washington, D.C. is a city in the United States, ranked 64th, with a population of 672,228 in 2015.
  Fact 0.64: Kingston is a city in Jamaica, ranked 65th, with a population of 669,627 in 2015.
  Fact 0.65: Boston is a city in the United States, ranked 66th, with a population of 667,137 in 2015.
  Fact 0.66: Tlaquepaque is a city in Mexico, ranked 67th, with a population of 664,193 in 2015.
  Fact 0.67: Memphis is a city in the United States, ranked 68th, with a population of 655,770 in 2015.
  Fact 0.68: Nashville is a city in the United States, ranked 69th, with a population of 654,610 in 2015.
  Fact 0.69: Torreón is a city in Mexico, ranked 70th, with a population of 653,150 in 2015.
  Fact 0.70: Vancouver is a city in Canada, ranked 71st, with a population of 648,608 in 2015.
  Fact 0.71: Portland is a city in the United States, ranked 72nd, with a population of 632,309 in 2015.
  Fact 0.72: Reynosa is a city in Mexico, ranked 73rd, with a population of 631,611 in 2015.
  Fact 0.73: Oklahoma City is a city in the United States, ranked 74th, with a population of 631,346 in 2015.
  Fact 0.74: Las Vegas is a city in the United States, ranked 75th, with a population of 623,747 in 2015.
  Fact 0.75: Baltimore is a city in the United States, ranked 76th, with a population of 621,849 in 2015.
  Fact 0.76: Brampton is a city in Canada, ranked 77th, with a population of 620,700 in 2016.
  Fact 0.77: Louisville is a city in the United States, ranked 78th, with a population of 615,366 in 2015.
  Fact 0.78: Morelia is a city in Mexico, ranked 79th, with a population of 608,190 in 2015.
  Fact 0.79: Milwaukee is a city in the United States, ranked 80th, with a population of 600,155 in 2015.
  Fact 0.80: Tuxtla Gutiérrez is a city in Mexico, ranked 81st, with a population of 598,710 in 2015.
  Fact 0.81: Apodaca is a city in Mexico, ranked 82nd, with a population of 597,207 in 2015.
  Fact 0.82: Durango is a city in Mexico, ranked 83rd, with a population of 590,893 in 2015.
  Fact 0.83: Villa Nueva is a city in Guatemala, ranked 84th, with a population of 564,686 in 2015.
  Fact 0.84: Albuquerque is a city in the United States, ranked 85th, with a population of 559,121 in 2015.
  Fact 0.85: Hamilton is a city in Canada, ranked 86th, with a population of 556,359 in 2015.
  Fact 0.86: Quebec City is a city in Canada, ranked 87th, with a population of 540,994 in 2015.
  Fact 0.87: Tonalá is a city in Mexico, ranked 88th, with a population of 536,111 in 2015.
  Fact 0.88: Tucson is a city in the United States, ranked 89th, with a population of 531,641 in 2015.
  Fact 0.89: Cuautitlán Izcalli is a city in Mexico, ranked 90th, with a population of 531,041 in 2015.
  Fact 0.90: Surrey is a city in Canada, ranked 91st, with a population of 526,004 in 2015.
  Fact 0.91: Ciudad López Mateos is a city in Mexico, ranked 92nd, with a population of 523,296 in 2015.
  Fact 0.92: Tultitlán is a city in Mexico, ranked 93rd, with a population of 520,557 in 2015.
  Fact 0.93: Fresno is a city in the United States, ranked 94th, with a population of 520,052 in 2015.
  Fact 0.94: Carrefour is a city in Haiti, ranked 95th, with a population of 501,768 in 2015.

Retrieved items:
<item 1>
List of North American cities by population
437,939 2015 15 San Diego United States 1,394,928 2015 16 Juárez Mexico 1,382,753 2015 17 León Mexico 1,349,224 2015 18 Dallas United States 1,300,092 2015 19
Tijuana Mexico 1,298,475 2015 20 Calgary Canada 1,230,915 2015 21 Tegucigalpa Honduras 1,190,230 2015 22 Zapopan Mexico 1,179,681 2015 23 Monterrey Mexico
1,109,171 2015 24 Managua Nicaragua 1,048,134 2015 25 Nezahualcóyotl Mexico 1,039,867 2015 26 San Jose United States 1,026,908 2015 27 Santo Domingo Dominican
Republic 1,007,997 2015 28 Guatemala City Guatemala 994,078 2015 29 Port - au - Prince Haiti 987,310 2015 30 Naucalpan Mexico 970,
</item 1>

<item 2>
The list ranks 95 North American cities with more than 500,000 residents within city limits, using mostly mid-2015 census figures, estimates, or projections; it
excludes metro-area populations. Mexico’s figures use the March 15, 2015 intercensal survey. Calgary uses an April 2015 municipal census; Edmonton uses its
April 2016 census (899,447). Mexico City leads with 8,918,653, followed by New York (8,550
</item 2>

<item 3>
List of North American cities by population
300 2015 48 San Luis Potosí Mexico 759,854 2015 49 Veracruz Mexico 752,171 2015 50 San Pedro Sula Honduras 742,118 2015 51 Santiago de los Caballeros Dominican
Republic 723,162 2015 52 Culiacán Mexico 720,798 2015 53 Winnipeg Canada 718,400 2015 54 Mexicali Mexico 717,628 2015 55 Cancún Mexico 711,682 2015 56 Acapulco
Mexico 710,261 2015 57 Tlalnepantla Mexico 700,734 2015 58 Seattle United States 684,451 2015 59 Guadalupe Mexico 682,880 2015 60 Denver United States 682,545
2015 61 El Paso United States 681,124 2015 62 Chimalhuacán Mexico 679,811 2015 63 Detroit United States 677,116 2015 64 Washington , D.C. United States 672,228
2015 65 Kingston Jamaica 669,
</item 3>

<item 4>
List of North American cities by population
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
List of North American cities by population
310 2015 30 Naucalpan Mexico 970,012 2015 31 Ottawa Canada 956,710 2015 32 Austin United States 931,830 2015 33 Edmonton Canada 899,447 2016 34 Mérida Mexico
892,363 2015 35 Querétaro Mexico 878,931 2015 36 Toluca Mexico 873,536 2015 37 Jacksonville United States 868,031 2015 38 Chihuahua Mexico 867,736 2015 39 San
Francisco United States 864,816 2015 40 Indianapolis United States 853,173 2015 41 Columbus United States 850,106 2015 42 Fort Worth United States 833,319 2015
43 Charlotte United States 827,097 2015 44 Hermosillo Mexico 819,999 2015 45 Saltillo Mexico 791,074 2015 46 Aguascalientes Mexico 785,942 2015 47 Mississauga
Canada 761,300 2015 48 San Luis Potosí Mexico 759,
</item 5>

<item 6>
List of North American cities by population
Retrieved 2016 - 05 - 01 . ( hide ) World 's largest cities City proper World ( Capitals ) Americas ( North Latin Central South ) Asia ( Arab world ASEAN Arabia
Far East ) Europe ( European Union Nordic ) Metropolitan area World Americas ( North South West Indies ) Europe ( European Union Nordic ) Asia Africa
Transborder Aggregate urban area † World Europe ( European Union Nordic ) Asia Urban agglomeration † North America Europe ( European Union ) Asia Historical
World Europe Related articles Global city Megacity Megalopolis Conurbation Suburbs † Aggregate urban areas are entire continuous built - up areas , grouping
together contiguous cities ( known as conurbations ) , whereas urban agglomerations are the built - up areas of individual cities . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_North_American_cities_by_population&oldid=822737139 '' Categories : Lists of cities by population Lists of
cities in the Americas North America - related lists Cities in North America Talk Contents About Wikipedia Español اردو 中文 Edit links This page was last edited
on 28 January 2018 , at 05 : 43 .
</item 6>

<item 7>
Beijing
: Doha 2021 : Eugene World 's twenty most populous metropolitan areas 1 Tokyo - Yokohama 2 Shanghai 3 Jakarta 4 Delhi 5 Seoul - Incheon 6 Karachi 7 Guangzhou 8
Beijing 9 Shenzhen 7 Mexico City 11 São Paulo 12 Lagos 13 Mumbai 14 Cairo 15 New York 16 Osaka 17 Moscow 18 Wuhan 19 Chengdu 20 Dhaka World 's fifty most -
populous urban areas Tokyo -- Yokohama ( Keihin ) Jakarta ( Jabodetabek ) Delhi Manila ( Metro Manila ) Seoul -- Incheon ( Sudogwon ) Shanghai Karachi Beijing
New York City Guangzhou -- Foshan ( Guangfo ) São Paulo Mexico City ( Valley of Mexico ) Mumbai Osaka -- Kobe -- Kyoto ( Keihanshin ) Moscow Dhaka Greater Cairo
Los Angeles Bangkok Kolkata Greater Buenos Aires Tehran Istanbul Lagos Shenzhen Rio de Janeiro Kinshasa Tianjin Paris Lima Chengdu Greater London Nagoya (
Chūkyō ) Lahore Chennai Bangalore Chicago Bogotá Ho Chi Minh City Hyderabad Dongguan Johannesburg Wuhan Taipei - Taoyuan Hangzhou Hong Kong Chongqing Ahmedabad
Kuala Lumpur ( Klang Valley ) Quanzhou VIAF
</item 7>

<item 8>
List of North American cities by population
List of North American cities by Population - wikipedia List of North American cities by Population Jump to : navigation , search For the majority of cities in
North America , the most recent official population census results , estimates or short - term projections date to 2015 . This list compiles figures for all
North American cities with a population within city limits exceeding 500,000 that year . These figures do not reflect the population of the urban agglomeration
or metropolitan area which typically do not coincide with the administrative boundaries of the city . They refer to mid-2015 populations with the following
exceptions : Mexican cities , whose figures derive from the 2015 Intercensal Survey conducted by INEGI with a reference date of 15 March 2015 ; Calgary , whose
2015 municipal census had a reference date of April 1 ; Edmonton , which conducted municipal censuses in April 2014 and April 2016 but has not reported an
intercensal estimate of its 2015 population . The April 2016 census result is given instead .
</item 8>

<item 9>
List of North American cities by population
The April 2016 census result is given instead . Contents ( hide ) 1 List 2 See also 3 Notes 4 References List ( edit ) City Image Country Population Year Mexico
City Mexico 8,918,653 2015 New York City United States 8,550,405 2015 Los Angeles United States 3,971,883 2015 Toronto Canada 2,826,498 2015 Chicago United
States 2,720,546 2015 6 Houston United States 2,296,224 2015 7 Havana Cuba 2,117,625 2015 8 Montreal Canada 1,753,034 2015 9 Ecatepec de Morelos Mexico
1,677,678 2015 10 Philadelphia United States 1,567,442 2015 11 Phoenix United States 1,563,025 2015 12 San Antonio United States 1,469,845 2015 13 Guadalajara
Mexico 1,460,148 2015 14 Puebla Mexico 1,437,939 2015 15 San Diego United States 1,394,
</item 9>

<item 10>
List of North American cities by population
C. United States 672,228 2015 65 Kingston Jamaica 669,627 2015 66 Boston United States 667,137 2015 67 Tlaquepaque Mexico 664,193 2015 68 Memphis United States
655,770 2015 69 Nashville United States 654,610 2015 70 Torreón Mexico 653,150 2015 71 Vancouver Canada 648,608 2015 72 Portland United States 632,309 2015 73
Reynosa Mexico 631,611 2015 74 Oklahoma City United States 631,346 2015 75 Las Vegas United States 623,747 2015 76 Baltimore United States 621,849 2015 77
Brampton Canada 620,700 2016 78 Louisville United States 615,366 2015 79 Morelia Mexico 608,190 2015 80 Milwaukee United States 600,155 2015 81 Tuxtla Gutiérrez
Mexico 598,710 2015 82 Apodaca Mexico 597,207 2015 83 Durango Mexico 590,
</item 10>
=== END ITEM id=13540d1ed48e43f7 ===

=== ITEM id=7923669cedbe5b1f ===
Gold sentences and facts:
Sentence 0: The first mention of the expression `` Apostles ' Creed '' occurs in a letter of AD 390 from a synod in Milan and may have been associated with the belief , widely accepted in the 4th century , that , under the inspiration of the Holy Spirit , each of the Twelve Apostles contributed an article of a creed .
  Fact 0.0: The first mention of the expression Apostles' Creed occurs in a letter of AD 390.
  Fact 0.1: The letter was from a synod in Milan.
  Fact 0.2: The expression Apostles' Creed may have been associated with a belief.
  Fact 0.3: The belief was widely accepted in the 4th century.
  Fact 0.4: The belief was that, under the inspiration of the Holy Spirit, each of the Twelve Apostles contributed an article of a creed.
Sentence 1: The title Symbolum Apostolicum ( Symbol or Creed of the Apostles ) appears for the first time in a letter , probably written by Ambrose , from a Council in Milan to Pope Siricius in about AD 390 `` Let them give credit to the Creed of the Apostles , which the Roman Church has always kept and preserved undefiled '' . But what existed at that time was not what is now known as the Apostles ' Creed but a shorter statement of belief that , for instance , did not include the phrase `` maker of heaven and earth '' , a phrase that may have been inserted only in the 7th century .
  Fact 1.0: The title Symbolum Apostolicum (Symbol or Creed of the Apostles) appears for the first time in a letter.
  Fact 1.1: The letter was probably written by Ambrose.
  Fact 1.2: The letter was from a Council in Milan to Pope Siricius.
  Fact 1.3: The letter dates to about AD 390.
  Fact 1.4: The letter says: Let them give credit to the Creed of the Apostles, which the Roman Church has always kept and preserved undefiled.
  Fact 1.5: What existed at that time was not what is now known as the Apostles' Creed.
  Fact 1.6: What existed at that time was a shorter statement of belief.
  Fact 1.7: That shorter statement did not include the phrase maker of heaven and earth, for instance.
  Fact 1.8: The phrase maker of heaven and earth may have been inserted only in the 7th century.

Retrieved items:
<item 1>
Apostles' Creed
Its basis appears to be the old Roman Creed known also as the Old Roman Symbol . Because of the early origin of its original form , it does not address some
Christological issues defined in the Nicene and other Christian Creeds . It thus says nothing explicitly about the divinity of either Jesus or the Holy Spirit .
This makes it acceptable to many Arians and Unitarians . Nor does it address many other theological questions which became objects of dispute centuries later .
The first mention of the expression `` Apostles ' Creed '' occurs in a letter of AD 390 from a synod in Milan and may have been associated with the belief ,
widely accepted in the 4th century , that , under the inspiration of the Holy Spirit , each of the Twelve Apostles contributed an article of a creed . Contents
( hide ) 1 Origin 2 Musical settings 3 Text in Latin 4 Text in Greek 5 English translations 5.1 Ecumenical version 5.2 Roman Catholic Church 5.2. 1 Articles 1
-- 12 5.2. 2 Articles 9 -- 10 5.3 Church of England 5.4 Lutheran Church 5.4.
</item 1>

<item 2>
Apostles' Creed
But what existed at that time was not what is now known as the Apostles ' Creed but a shorter statement of belief that , for instance , did not include the
phrase `` maker of heaven and earth '' , a phrase that may have been inserted only in the 7th century . This illumination from a 13th - century manuscript shows
the apostles writing the Creed , receiving inspiration from the Holy Spirit . The account of the origin of this creed , the forerunner and principal source of
the Apostles ' Creed , as having been jointly created by the Apostles under the inspiration of the Holy Spirit , with each of the twelve contributing one of the
articles , was already current at that time . The earlier text evolved from simpler texts based on Matthew 28 : 19 , part of the Great Commission , and it has
been argued that it was already in written form by the late 2nd century ( c. 180 ) .
</item 2>

<item 3>
Apostles' Creed
180 ) . While the individual statements of belief that are included in the Apostles ' Creed -- even those not found in the Old Roman Symbol -- are found in
various writings by Irenaeus , Tertullian , Novatian , Marcellus , Rufinus , Ambrose , Augustine , Nicetas , and Eusebius Gallus , the earliest appearance of
what we know as the Apostles ' Creed was in the De singulis libris canonicis scarapsus ( `` Excerpt from Individual Canonical Books '' ) of St. Pirminius (
Migne , Patrologia Latina 89 , 1029 ff . ) , written between 710 and 714 . Bettenson and Maunder state that it is first from Dicta Abbatis Pirminii de singulis
libris canonicis scarapsus ( idem quod excarpsus , excerpt ) , c. 750 . This longer Creed seems to have arisen in what is now France and Spain . Charlemagne
imposed it throughout his dominions , and it was finally accepted in Rome , where the Old Roman Symbol or similar formulas had survived for centuries .
</item 3>

<item 4>
Apostles' Creed
It has been argued nonetheless that it dates from the second half of the 5th century , though no earlier . Some have suggested that the Apostles ' Creed was
spliced together with phrases from the New Testament . For instance , the phrase `` descendit ad inferos '' ( `` he descended into hell '' ) echoes Ephesians 4
: 9 , `` κατέβη εἰς τὰ κατώτερα μέρη τῆς γῆς '' ( `` he descended into the lower , earthly regions '' ) . It is of interest that this phrase first appeared in
one of the two versions of Rufinus in AD 390 and then did not appear again in any version of the creed until AD 650 . This phrase and that on the communion of
saints are articles found in the Apostles ' Creed , but not in the Old Roman Symbol nor in the Nicene Creed . Musical settings ( edit ) Musical settings of the
Symbolum Apostolorum as a motet are rare .
</item 4>

<item 5>
The Apostles’ Creed (also called the Symbol of the Apostles) is an early, Trinitarian statement of Christian belief, structured around God the Father, Jesus
Christ, and the Holy Spirit. Based on the canonical Gospels and New Testament letters, and rooted in the Old Roman Creed, it is widely used in Western Christian
worship and teaching, including by Catholics, Anglicans, Lutherans, Presbyterians, Methodists, and Congregationalists. Its title appears in a letter
</item 5>

<item 6>
Apostles' Creed
Apostles ' Creed - wikipedia Apostles ' Creed Jump to : navigation , search The rubric above this 13th - century illuminated manuscript translates `` twelve
articles of faith set out by twelve apostles '' . The Apostles ' Creed ( Latin : Symbolum Apostolorum or Symbolum Apostolicum ) , sometimes entitled Symbol of
the Apostles , is an early statement of Christian belief -- a creed or `` symbol '' . It is widely used by a number of Christian denominations for both
liturgical and catechetical purposes , most visibly by liturgical Churches of Western tradition , including the Catholic Church , Lutheranism and Anglicanism .
It is also used by Presbyterians , Methodists and Congregationalists . The Apostles ' Creed is Trinitarian in structure with sections affirming belief in God
the Father , Jesus Christ His Son and the Holy Spirit . The Apostles ' Creed was based on Christian theological understanding of the Canonical gospels , the
letters of the New Testament and to a lesser extent the Old Testament . Its basis appears to be the old Roman Creed known also as the Old Roman Symbol .
</item 6>

<item 7>
Apostles' Creed
See also ( edit ) Christianity portal Nicene Creed Chalcedonian Creed Lutheran Book of Worship Virgin birth of Jesus Passion of Jesus One , Holy , Catholic ,
and Apostolic Church Notes ( edit ) Jump up ^ Not in the sense that the word `` symbol '' has in modern English , but in the original meaning of the word ,
derived from `` Latin symbolum , sign , token , from Greek σύμβολον , token for identification ( by comparing with its counterpart ) , from συμβάλλειν , to
throw together , compare '' . Jump up ^ In a suggested revision of 1923 , this line has `` of the Holy Ghost '' . The Scottish Book of Common Prayer ( 1637 )
has `` which was conceived of the holy Ghost '' . This goes back to the 1559 Book of Common Prayer , which has `` whiche was conceived of the holy Ghost '' .
Jump up ^ This ecumenical version of the English Language Liturgical Consultation ( ELLC ) and its predecessor .
</item 7>

<item 8>
The Apostles’ Creed developed from the Old Roman Creed, an early statement of faith likely written by about AD 180 and rooted in Matthew 28:19. Its original
form was shorter than today’s creed; “maker of heaven and earth” may have been added in the 7th century. It does not explicitly affirm Jesus’ or the Holy
Spirit’s divinity, making it acceptable to many Arians and Unitarians. The name first appears in a letter from the Synod of Milan
</item 8>

<item 9>
Apostles' Creed
1 Evangelical Lutheran Worship 5.4. 2 Lutheran Service Book 5.4. 3 Church of Denmark 5.5 United Methodist Church 6 Liturgical use in Western Christianity 6.1
Catholic Church 6.1. 1 Rite of Baptism 6.1. 2 Profession of faith at Mass 6.1. 3 Use in the Divine Office 6.2 Anglican Communion 6.2. 1 Episcopal Church (
United States ) 6.3 Lutheran Church 6.4 Methodism 7 See also 8 Notes 9 References 10 Further reading 11 External links 11.1 English translations Origin ( edit )
The title Symbolum Apostolicum ( Symbol or Creed of the Apostles ) appears for the first time in a letter , probably written by Ambrose , from a Council in
Milan to Pope Siricius in about AD 390 `` Let them give credit to the Creed of the Apostles , which the Roman Church has always kept and preserved undefiled ''
.
</item 9>

<item 10>
The Apostles’ Creed is an early Western Christian statement of Trinitarian belief, derived from the Old Roman Creed, likely formed by about AD 180. Its wording
developed over centuries, with a complete form documented in the early eighth century. It affirms God the Father and creator; Jesus’ conception, death, descent
to the dead or hell, resurrection, ascension, and judgment; and the Holy Spirit, Church, communion of saints, forgiveness, bodily resurrection, and eternal
life. Catholics
</item 10>
=== END ITEM id=7923669cedbe5b1f ===

=== ITEM id=84ed308a7c15635d ===
Gold sentences and facts:
Sentence 0: Health or vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually rewarded in some way .
  Fact 0.0: Health or vitality is an attribute assigned to entities in a role-playing or video game.
  Fact 0.1: Health or vitality is assigned to entities such as the player character, enemies, and objects.
  Fact 0.2: Health or vitality indicates the state of an entity in combat.
  Fact 0.3: Health is usually measured in hit points or health points.
  Fact 0.4: Hit points or health points are shortened to HP.
  Fact 0.5: When the HP of a player character reaches zero, the player may lose a life.
  Fact 0.6: When the HP of a player character reaches zero, the character might become incapacitated or die.
  Fact 0.7: When the HP of an enemy reaches zero, the enemy may be defeated or die.
  Fact 0.8: When the HP of an enemy reaches zero, the player is usually rewarded in some way.

Retrieved items:
<item 1>
Harry Potter (film series)
19 November 2010 . Retrieved 30 September 2012 . Jump up ^ `` A Wizard Comes Of Age '' . Retrieved 29 September 2012 . ^ Jump up to : `` Kodak Celebrates the
Oscars ® Feature : Harry Potter and the Deathly Hallows -- Part 1 '' . Kodak . Archived from the original on 25 May 2013 . Retrieved 29 September 2012 . Jump up
^ `` Deathly Hallows to Be Shot Using `` Loads of Hand - Held Cameras , '' Tom Felton Talks Sectumsempra in Half - Blood Prince `` . The Leaky Cauldron. 31
March 2009 . Retrieved 31 March 2009 . Jump up ^ `` Lensers are n't afraid of the dark '' . Variety . 12 December 2012 . Retrieved 30 September 2012 . What I
loved about the last film is that David pushed me to go dark , which all cinematographers love to do . And usually you 're fighting with the producers ( about
the look ) but they all wanted it dark and atmospheric , too .
</item 1>

<item 2>
Harry Potter (film series)
Campaignlive.co.uk . Retrieved 19 October 2017 . ^ Jump up to : Dargis , Manohla ; Scott , A.O. ( 15 July 2007 ) . `` Harry Potter and the Four Directors '' .
The New York Times . Retrieved 29 July 2011 . ^ Jump up to : `` Harry Potter at Leavesden '' . WB Studio Tour . Archived from the original on 10 February 2014 .
Retrieved 16 September 2012 . Jump up ^ `` Warner Bros. Plans Two - Part Film Adaptation of `` Harry Potter and the Deathly Hallows '' to Be Directed by David
Yates `` . Business Wire. 13 March 2008 . Retrieved 6 September 2012 ... expand the screen adaptation of Harry Potter and the Deathly Hallows and release the
film in two parts . Jump up ^ Boucher , Geoff ; Eller , Claudia ( 7 November 2010 ) . `` The end nears for ' Harry Potter ' on film '' . Los Angeles Times .
Retrieved 3 January 2010 .
</item 2>

<item 3>
Harry Potter (film series)
The fifth film , Harry Potter and the Order of the Phoenix , was released by Warner Bros. on 11 July 2007 in English - speaking countries , except for the UK
and Ireland , which released the movie on 12 July . The sixth , Harry Potter and the Half - Blood Prince , was released on 15 July 2009 to critical acclaim and
finished its theatrical run ranked as the number two grossing film of 2009 on the worldwide charts . The final novel , Harry Potter and the Deathly Hallows ,
was split into two cinematic parts : Part 1 was released on 19 November 2010 , and Part 2 , the conclusion to both the final film and the series , was released
on 15 July 2011 . Part 1 was originally scheduled to be released in 3D and 2D , but due to a delay in the 3D conversion process , Warner Bros. released the film
only in 2D and IMAX cinemas . However , Part 2 was released in 2D and 3D cinemas as originally planned .
</item 3>

<item 4>
Harry Potter (film series)
In 2006 , Nicholas Hooper started work on the score to Order of the Phoenix by reuniting with director David Yates . Hooper also composed the soundtrack to Half
- Blood Prince but decided not to return for the final films . In January 2010 , Alexandre Desplat was confirmed to compose the score for Harry Potter and the
Deathly Hallows -- Part 1 . The film 's orchestration started in the summer with Conrad Pope , the orchestrator on the first three Harry Potter films ,
collaborating with Desplat . Pope commented that the music `` reminds one of the old days . '' Desplat returned to score Harry Potter and the Deathly Hallows --
Part 2 in 2011 . Director David Yates stated that he wanted John Williams to return to the series for the final installment , but their schedules did not align
due to the urgent demand for a rough cut of the film . The final recording sessions of Harry Potter took place on 27 May 2011 at Abbey Road Studios with the
London Symphony Orchestra , orchestrator Conrad Pope , and composer Alexandre Desplat .
</item 4>

<item 5>
Health (gaming)
In video games , health can also be displayed graphically , such as with a bar that empties itself when an entity loses health ( a health bar , typically red )
, icons that are `` chipped away '' , or in more novel ways . Contents ( hide ) 1 History 2 Usage 2.1 Regeneration 3 Display 4 References History ( edit )
Dungeons & Dragons co-creator Dave Arneson described the origin of hit points in a 2002 interview . When Arneson was adapting the medieval wargame Chainmail (
1971 ) to a fantasy setting , a process that with Gary Gygax would lead to Dungeons & Dragons , he saw that the emphasis of the gameplay was moving from large
armies to small groups of heroes and eventually to the identification of one player and one character that is essential to role - playing as it was originally
conceived . Players became attached to their heroes and did not want them to die every time they lost a die roll . Players were thus given multiple hit points
which were incrementally decreased as they took damage . Arneson took the concept , along with armor class , from a set of a naval American Civil War game 's
rules .
</item 5>

<item 6>
Health (gaming)
US Navy used similar concept in their tactical war games already in 1920s and 1930s . In their simulation each ship had Life parameter . The unit of Life of the
ship was a number of `` equivalent penetrative 14 - inch shell hits '' . The Navy considered ie. that Kongō - class battlecruiser had 12 Life points and Nagato
- class battleship had 18.8 . A visual power meter representing stamina was used in Nintendo 's 1983 arcade game Punch - Out ! ! and Data East 's 1981 DECO
Cassette System arcade game Flash Boy . Usage ( edit ) In action video games as well as in role - playing games , health points can usually be depleted by
attacking the entity . A defense attribute might reduce the amount of HP that is lost when a character is damaged . It is common in role - playing games for a
character 's maximum health and defense attributes to be gradually raised as the character levels up . In game design , it is deemed important that a player is
aware of it when they are losing health , each hit playing a clear sound effect .
</item 6>

<item 7>
Harry Potter (film series)
Harry Potter and the Deathly Hallows -- Part 2 ( 2011 ) Main article : Harry Potter and the Deathly Hallows -- Part 2 After destroying one Horcrux and
discovering the significance of the three Deathly Hallows , Harry , Ron and Hermione continue to seek the other Horcruxes in an attempt to destroy Voldemort ,
who has now obtained the powerful Elder Wand . The Dark Lord discovers Harry 's hunt for Horcruxes and launches an attack on Hogwarts School , where the trio
return for one last stand against the dark forces that threaten the Wizarding and Muggle worlds . Release The rights for the first four novels in the series
were sold to Warner Bros. for £ 1,000,000 by J.K. Rowling . After the release of the fourth book in July 2000 , the first film , Harry Potter and the
Philosopher 's Stone , was released on 16 November 2001 . In its opening weekend , the film grossed $90 million in the United States alone , which set a record
opening worldwide . The succeeding three motion picture adaptations followed suit in financial success , while garnering positive reviews from fans and critics
.
</item 7>

<item 8>
Health (gaming)
Health ( Gaming ) - wikipedia Health ( Gaming ) Jump to : navigation , search A health bar , a possible representation of the health of a character . Health or
vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its
state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may
lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually
rewarded in some way . Any entity within a game could have a health value , including the player character , non-player characters and objects . Indestructible
entities have no diminishable health value . Health might be displayed as a numeric value , such as `` 50 / 100 '' . Here , the first number indicates the
current amount of HP an entity has and the second number indicates the entity 's maximum HP .
</item 8>

<item 9>
Harry Potter (film series)
Extraordinary images , extraordinary talent '' . Framestore . Jump up ^ `` Cinesite , HP 1 -- 7 '' . Cinesite.com . Archived from the original on 18 December
2010 . Retrieved 2 March 2011 . Jump up ^ `` June 2010 Filming completed '' . Snitchseeker.com. 12 June 2010 . Retrieved 2 March 2011 . Jump up ^ `` 10 Years
Filming -- 24Dec2010 '' . Snitchseeker.com. 21 December 2010 . Retrieved 2 March 2011 . Jump up ^ MuggleNet ( 21 December 2010 ) . `` JK Title reveal '' .
Mugglenet.com . Retrieved 2 March 2011 . Jump up ^ `` Harry Potter and the Order of the Phoenix ( 2007 ) '' . IMDb.com . Jump up ^ `` WB Name Drops Big Titles
'' . ERCBoxOffice. 23 February 2009 . Archived from the original on 26 February 2009 .
</item 9>

<item 10>
Harry Potter (film series)
Jump up ^ MuggleNet ( 19 January 2010 ) . `` CONFIRMED -- Desplat for DH '' . Mugglenet.com . Retrieved 2 March 2011 . Jump up ^ `` Pope on Desplat 's HP7 Pt. 1
Score '' . Snitchseeker.com. 28 August 2010 . Retrieved 2 March 2011 . Jump up ^ `` Alexandre Desplat -- Composer of Part 1 and 2 ( NOTE : Click `` About the
Movie '' , then `` Filmmakers '' , then `` Alexandre Desplat '' ) `` . Harrypotter.warnerbros.com . Retrieved 2 March 2011 . Jump up ^ MuggleNet ( 12 November
2010 ) . `` Yates on Williams , Part 2 '' . Mugglenet.com . Retrieved 2 March 2011 . Jump up ^ `` Conrad Pope '' . www.facebook.com . Jump up ^ `` Framestore -
We are Framestore . Extraordinary images , extraordinary talent '' . Framestore .
</item 10>
=== END ITEM id=84ed308a7c15635d ===

=== ITEM id=14c49ed84c72faa6 ===
Gold sentences and facts:
Sentence 0: Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24 Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5 Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia : 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1
  Fact 0.0: Spain is listed with the value 430.
  Fact 0.1: France is listed with the value 128.
  Fact 0.2: Italy is listed with the value 104.
  Fact 0.3: Russia is listed with the value 100.
  Fact 0.4: Portugal is listed with the value 87.
  Fact 0.5: Germany is listed with the value 77.
  Fact 0.6: United Kingdom is listed with the value 64.
  Fact 0.7: Poland is listed with the value 47.
  Fact 0.8: Greece is listed with the value 46.
  Fact 0.9: Turkey is listed with the value 41.
  Fact 0.10: Belgium is listed with the value 31.
  Fact 0.11: Netherlands is listed with the value 28.
  Fact 0.12: Romania is listed with the value 24.
  Fact 0.13: Switzerland is listed with the value 20.
  Fact 0.14: Austria is listed with the value 14.
  Fact 0.15: Sweden is listed with the value 12.
  Fact 0.16: Croatia is listed with the value 10.
  Fact 0.17: Ireland is listed with the value 9.
  Fact 0.18: Ukraine is listed with the value 9.
  Fact 0.19: Hungary is listed with the value 8.
  Fact 0.20: Bulgaria is listed with the value 6.
  Fact 0.21: Cyprus is listed with the value 6.
  Fact 0.22: Czech Republic is listed with the value 6.
  Fact 0.23: Finland is listed with the value 5.
  Fact 0.24: Lithuania is listed with the value 5.
  Fact 0.25: Norway is listed with the value 5.
  Fact 0.26: Serbia is listed with the value 5.
  Fact 0.27: Slovenia is listed with the value 5.
  Fact 0.28: Denmark is listed with the value 4.
  Fact 0.29: Latvia is listed with the value 4.
  Fact 0.30: Estonia is listed with the value 3.
  Fact 0.31: Azerbaijan is listed with the value 3.
  Fact 0.32: Bosnia and Herzegovina is listed with the value 3.
  Fact 0.33: Georgia is listed with the value 3.
  Fact 0.34: Luxembourg is listed with the value 3.
  Fact 0.35: Slovakia is listed with the value 3.
  Fact 0.36: Armenia is listed with the value 2.
  Fact 0.37: Albania is listed with the value 1.
  Fact 0.38: Andorra is listed with the value 1.
  Fact 0.39: Belarus is listed with the value 1.
  Fact 0.40: Iceland is listed with the value 1.
  Fact 0.41: Macedonia is listed with the value 1.
  Fact 0.42: Malta is listed with the value 1.
  Fact 0.43: Monaco is listed with the value 1.
  Fact 0.44: Montenegro is listed with the value 1.
Sentence 1: United Kingdom : 64
  Fact 1.0: United Kingdom is listed with the value 64.
Sentence 2: Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32 Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20 Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait : 5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24 Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5 Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia : 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1
  Fact 2.0: South Africa is listed under Africa with the value 9.
  Fact 2.1: Egypt is listed under Africa with the value 5.
  Fact 2.2: Morocco is listed under Africa with the value 4.
  Fact 2.3: Tunisia is listed under Africa with the value 3.
  Fact 2.4: Algeria is listed under Africa with the value 1.
  Fact 2.5: United States is listed under Americas with the value 85.
  Fact 2.6: Mexico is listed under Americas with the value 84.
  Fact 2.7: Brazil is listed under Americas with the value 56.
  Fact 2.8: Canada is listed under Americas with the value 32.
  Fact 2.9: Colombia is listed under Americas with the value 14.
  Fact 2.10: Argentina is listed under Americas with the value 10.
  Fact 2.11: Chile is listed under Americas with the value 9.
  Fact 2.12: Venezuela is listed under Americas with the value 8.
  Fact 2.13: Peru is listed under Americas with the value 4.
  Fact 2.14: Puerto Rico is listed under Americas with the value 4.
  Fact 2.15: Dominican Republic is listed under Americas with the value 3.
  Fact 2.16: Guatemala is listed under Americas with the value 3.
  Fact 2.17: Panama is listed under Americas with the value 3.
  Fact 2.18: Costa Rica is listed under Americas with the value 2.
  Fact 2.19: Ecuador is listed under Americas with the value 2.
  Fact 2.20: El Salvador is listed under Americas with the value 2.
  Fact 2.21: Honduras is listed under Americas with the value 2.
  Fact 2.22: Uruguay is listed under Americas with the value 2.
  Fact 2.23: Aruba is listed under Americas with the value 1.
  Fact 2.24: Nicaragua is listed under Americas with the value 1.
  Fact 2.25: Paraguay is listed under Americas with the value 1.
  Fact 2.26: China is listed under Asia with the value 200.
  Fact 2.27: Japan is listed under Asia with the value 99.
  Fact 2.28: South Korea is listed under Asia with the value 43.
  Fact 2.29: Saudi Arabia is listed under Asia with the value 39.
  Fact 2.30: Israel is listed under Asia with the value 24.
  Fact 2.31: India is listed under Asia with the value 20.
  Fact 2.32: Indonesia is listed under Asia with the value 17.
  Fact 2.33: Hong Kong is listed under Asia with the value 14.
  Fact 2.34: Thailand is listed under Asia with the value 11.
  Fact 2.35: United Arab Emirates is listed under Asia with the value 11.
  Fact 2.36: Malaysia is listed under Asia with the value 10.
  Fact 2.37: Singapore is listed under Asia with the value 9.
  Fact 2.38: Taiwan is listed under Asia with the value 9.
  Fact 2.39: Philippines is listed under Asia with the value 8.
  Fact 2.40: Lebanon is listed under Asia with the value 7.
  Fact 2.41: Kazakhstan is listed under Asia with the value 5.
  Fact 2.42: Kuwait is listed under Asia with the value 5.
  Fact 2.43: Qatar is listed under Asia with the value 5.
  Fact 2.44: Jordan is listed under Asia with the value 3.
  Fact 2.45: Bahrain is listed under Asia with the value 2.
  Fact 2.46: Macau is listed under Asia with the value 2.
  Fact 2.47: Oman is listed under Asia with the value 1.
  Fact 2.48: Vietnam is listed under Asia with the value 1.
  Fact 2.49: Spain is listed under Europe with the value 430.
  Fact 2.50: France is listed under Europe with the value 128.
  Fact 2.51: Italy is listed under Europe with the value 104.
  Fact 2.52: Russia is listed under Europe with the value 100.
  Fact 2.53: Portugal is listed under Europe with the value 87.
  Fact 2.54: Germany is listed under Europe with the value 77.
  Fact 2.55: United Kingdom is listed under Europe with the value 64.
  Fact 2.56: Poland is listed under Europe with the value 47.
  Fact 2.57: Greece is listed under Europe with the value 46.
  Fact 2.58: Turkey is listed under Europe with the value 41.
  Fact 2.59: Belgium is listed under Europe with the value 31.
  Fact 2.60: Netherlands is listed under Europe with the value 28.
  Fact 2.61: Romania is listed under Europe with the value 24.
  Fact 2.62: Switzerland is listed under Europe with the value 20.
  Fact 2.63: Austria is listed under Europe with the value 14.
  Fact 2.64: Sweden is listed under Europe with the value 12.
  Fact 2.65: Croatia is listed under Europe with the value 10.
  Fact 2.66: Ireland is listed under Europe with the value 9.
  Fact 2.67: Ukraine is listed under Europe with the value 9.
  Fact 2.68: Hungary is listed under Europe with the value 8.
  Fact 2.69: Bulgaria is listed under Europe with the value 6.
  Fact 2.70: Cyprus is listed under Europe with the value 6.
  Fact 2.71: Czech Republic is listed under Europe with the value 6.
  Fact 2.72: Finland is listed under Europe with the value 5.
  Fact 2.73: Lithuania is listed under Europe with the value 5.
  Fact 2.74: Norway is listed under Europe with the value 5.
  Fact 2.75: Serbia is listed under Europe with the value 5.
  Fact 2.76: Slovenia is listed under Europe with the value 5.
  Fact 2.77: Denmark is listed under Europe with the value 4.
  Fact 2.78: Latvia is listed under Europe with the value 4.
  Fact 2.79: Estonia is listed under Europe with the value 3.
  Fact 2.80: Azerbaijan is listed under Europe with the value 3.
  Fact 2.81: Bosnia and Herzegovina is listed under Europe with the value 3.
  Fact 2.82: Georgia is listed under Europe with the value 3.
  Fact 2.83: Luxembourg is listed under Europe with the value 3.
  Fact 2.84: Slovakia is listed under Europe with the value 3.
  Fact 2.85: Armenia is listed under Europe with the value 2.
  Fact 2.86: Albania is listed under Europe with the value 1.
  Fact 2.87: Andorra is listed under Europe with the value 1.
  Fact 2.88: Belarus is listed under Europe with the value 1.
  Fact 2.89: Iceland is listed under Europe with the value 1.
  Fact 2.90: Macedonia is listed under Europe with the value 1.
  Fact 2.91: Malta is listed under Europe with the value 1.
  Fact 2.92: Monaco is listed under Europe with the value 1.
  Fact 2.93: Montenegro is listed under Europe with the value 1.
  Fact 2.94: Australia is listed under Oceania with the value 19.
  Fact 2.95: New Zealand is listed under Oceania with the value 1.

Retrieved items:
<item 1>
Zara (retailer)
Ortega opened additional stores throughout Spain . During the 1980s , Ortega changed the design , manufacturing , and distribution process to reduce lead times
and react to new trends in a quicker way , which he called `` instant fashions '' . The improvements included the use of information technologies and using
groups of designers instead of individuals . Zara 's previous logo Expansion ( edit ) In 1988 , the company started its international expansion through Porto ,
Portugal . In 1989 , it entered the United States , and then France in 1990 . During the 1990s , Zara expanded to Mexico ( 1992 ) , Greece , Belgium and Sweden
( 1993 ) . In the early 2000s , Zara opened its first stores in Japan and Singapore ( 2002 ) , Russia and Malaysia ( 2003 ) , China , Morocco , Estonia ,
Hungary and Romania ( 2004 ) , the Philippines , Costa Rica and Indonesia ( 2005 ) , South Korea ( 2008 ) , India ( 2010 ) , and South Africa and Australia (
2011 ) .
</item 1>

<item 2>
Zara (retailer)
When the news was eventually picked up by media outlets , and Inditex , Zara 's owning company , was asked to comment on the issue , Inditex replied that the
items in question have been suspended from sale , and that they are in contact with Bassen 's lawyer to clarify and address the issue . In November 2017 ,
customers shopping at Zara stores in Istanbul , Turkey found handwritten notes purportedly from Turkish workers in the pockets of in - store garments asking
shoppers to pressure Zara into paying them the wages they claim they are owed . Stores ( edit ) There are over 2,100 Zara stores located across 88 countries .
Zara usually selects the best located and most expensive real estate locations in the world to open its flagship stores . Zara 's flagship stores : Fifth Avenue
, New York Oxford Street , London Calle Serrano , Madrid Via del Corso , Rome Champs - Élysées , Paris Oscar Freire Street , Sao Paulo Corso Vittorio Emanuele ,
Milan Nevsky Prospect , Saint Petersburg Shibuya and Ginza , Tokyo Myeongdong in Seoul Flora Fountain in Mumbai The number of Zara stores in each country ,
</item 2>

<item 3>
Zara (retailer)
: 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24
Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5
Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia
: 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1 See also ( edit )
Spain portal Companies portal Fashion portal Bonds ( clothing ) Gap Inc . H&M List of companies of Spain European Retail Round Table References ( edit ) Jump up
^ `` Zara '' . inditex.com .
</item 3>

<item 4>
Zara (retailer)
Zara ( retailer ) - Wikipedia Zara ( retailer ) Jump to : navigation , search Zara España , S.A. Type Sociedad Anónima Industry Retail Founded Arteixo , Galicia
( 24 May 1974 ; 43 years ago ( 1974 - 05 - 24 ) ) Founder Amancio Ortega Headquarters Arteixo , Galicia Number of locations 2266 stores Area served Worldwide
Products Clothing Revenue US $ 15.9 billion ( 2016 ) Parent Inditex Website zara.com Zara ( Spanish : ( ˈθaɾa ) ) is a Galician fast fashion ( clothing and
accessories ) retailer based in Arteixo , Galicia . The company was founded in 1975 by Amancio Ortega and Rosalía Mera . It is the main brand of the Inditex
group , the world 's largest apparel retailer . The fashion group also owns brands such as Massimo Dutti , Pull&Bear , Bershka , Stradivarius , Oysho , Zara
Home , and Uterqüe .
</item 4>

<item 5>
Zara (retailer)
24 April 2005 . Retrieved 8 April 2016 . Jump up ^ `` Spains Zara Opens First Store in Korea '' . The Korea Times . 29 April 2008 . Retrieved 8 April 2016 .
Jump up ^ `` Fashion chain Zara opens its first Indian store '' . BBC . 31 May 2010 . Retrieved 8 April 2016 . Jump up ^ Emma Jordan ( 19 January 2011 ) . ``
Zara to Open in South Africa '' . iFashion . Retrieved 8 April 2016 . Jump up ^ `` Global phenomenon Zara finally opens in Sydney '' . Vogue. 19 April 2011 .
Retrieved 8 April 2016 . Jump up ^ `` Zara launches online boutique '' . fashionunited.com. 6 September 2010 . Retrieved 22 June 2014 . Jump up ^ `` Zara.com
launches in 5 more countries '' . Retrieved 30 May 2015 . Jump up ^ `` Inditex post record profits '' .
</item 5>

<item 6>
Zara (retailer)
Products ( edit ) Zara stores have men 's and women 's clothing , as well as children 's clothing ( Zara Kids ) . Zara 's products are supplied based on
consumer trends . Its highly responsive supply chain ships new products to stores twice a week . After products are designed , they take ten to fifteen days to
reach the stores . All of the clothing is processed through the distribution center in Spain . New items are inspected , sorted , tagged , and loaded into
trucks . In most cases , the clothing is delivered within 48 hours . Zara produces over 450 million items per year . Manufacturing and distribution ( edit )
Zara Almere - Heren The men 's department of a typical Zara store . Almere , Netherlands Zara clothing made in Portugal Zara store in New York City . Zara store
in Brisbane Zara store in Palma de Mallorca ( Spain ) . Zara store in Brussels ( Belgium ) . Reportedly , Zara needs just one week to develop a new product and
get it to stores , compared to the six - month industry average , and launches around 12,000 new designs each year .
</item 6>

<item 7>
Zara (retailer)
20 October 2014 . Graham Ruddick . accessed 5 April 2016 Jump up ^ Zara new season styles : what to buy at the Spanish giant right now . Linda Sharkey. 29 May
2015 . Independent . accessed 5 April 2016 Jump up ^ `` Presencia internacional '' . inditex.com. 30 April 2015 . Retrieved 9 September 2015 . Jump up ^ ``
Localizador de tiendas '' ( in Spanish ) . Zara.com . Retrieved 11 June 2015 . External links ( edit ) Official website Media related to Zara ( clothing ) at
Wikimedia Commons Retrieved from `` https://en.wikipedia.org/w/index.php?title=Zara_(retailer)&oldid=819154896 '' Categories : Clothing brands of Spain Clothing
retailers of Spain Companies based in Galicia Clothing companies established in 1974 Retail companies established in 1974 1974 establishments in Spain Inditex
brands Hidden categories : Pages using citations with accessdate and no URL CS1 Dutch - language sources ( nl ) CS1 Portuguese - language sources ( pt )
</item 7>

<item 8>
Zara (retailer)
Zara as of 2017 manages up to 20 clothing collections a year . Contents ( hide ) 1 History 1.1 Early history 1.2 Expansion 2 Products 3 Manufacturing and
distribution 3.1 Non-toxic clothing 4 Controversy 5 Stores 6 See also 7 References 8 External links History ( edit ) Early history ( edit ) Amancio Ortega
opened the first Zara store in 1975 in downtown A Coruña , Galicia , Spain . Ortega initially named the store Zorba after the classic film Zorba the Greek , but
after learning there was a bar with the same name two blocks away , they rearranged the letters molded for the sign to ' Zara . ' It is believed the extra ' a '
came from an additional set of letters that had been made for the company . The first store featured low - priced lookalike products of popular , higher - end
clothing fashions . Ortega opened additional stores throughout Spain .
</item 8>

<item 9>
Zara (retailer)
On September 2010 , Zara launched its online boutique . The website began in Spain , the UK , Portugal , Italy , Germany and France . In November that same year
, Zara Online extended the service to five more countries : Austria , Ireland , the Netherlands , Belgium and Luxembourg . Online stores began operating in the
United States in 2011 , Russia and Canada in 2013 , and Mexico , Romania , and South Korea in 2014 . . India In 4 october 2017 . Zara introduced the use of RFID
technology in its stores in 2014 . The RFID chips are located in the security tags which are removed from clothing when it is purchased and can be reused . The
chip allows the company to quickly take inventory by detecting radio signals from the RFID tags . When an item is sold , the stockroom is immediately notified
so that the item can be replaced . An item that is not on the shelf can easily be found with the RFID tag . In 2015 , Zara was ranked 30 on Interbrand 's list
of best global brands .
</item 9>

<item 10>
Zara (retailer)
as of 31 November 2017 : Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32
Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El
Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20
Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait :
5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United
</item 10>
=== END ITEM id=14c49ed84c72faa6 ===

=== ITEM id=0b725f4689002f20 ===
Gold sentences and facts:
Sentence 0: Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother . Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New '' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti .
  Fact 0.0: Tracy McConnell is better known as 'The Mother'.
  Fact 0.1: Tracy McConnell is the title character from the CBS television sitcom How I Met Your Mother.
  Fact 0.2: The sitcom is narrated by Future Ted.
  Fact 0.3: The sitcom tells the story of how Ted Mosby met The Mother.
  Fact 0.4: Tracy McConnell appears in 8 episodes.
  Fact 0.5: Tracy McConnell's episodes run from 'Lucky Penny' to 'The Time Travelers'.
  Fact 0.6: Tracy McConnell appears as an unseen character in those episodes.
  Fact 0.7: Tracy McConnell was first seen fully in 'Something New'.
  Fact 0.8: Tracy McConnell was promoted to a main character in season 9.
  Fact 0.9: The Mother is played by Cristin Milioti.

Retrieved items:
<item 1>
The Mother (How I Met Your Mother)
While this incident made Cindy realise that she is a lesbian , it also made the Mother decide to go back into dating , as the kiss was her first in a long time
. Some time after this , a man named Darren approaches The Mother and is welcomed into her band named Superfreakonomics . Darren gradually takes over the band .
In the season 6 opener `` Big Days '' it is revealed Ted meets his future wife `` the day of '' the wedding at which he is the best man . In the episode ``
False Positive '' Robin asks Ted to be her future best man , should she ever get married . In the episode `` Challenge Accepted '' , it is revealed that Ted
meets the mother of his children the day of Barney 's wedding . In the last episode of season 7 , `` The Magician 's Code '' it is shown that Barney will marry
Robin , and Ted will meet the mother `` the day of '' their wedding . On the premiere of season 8 , Ted 's wife appears after Barney and Robin 's wedding ,
outside at the `` Farhampton '' station while holding a yellow umbrella and her bass guitar .
</item 1>

<item 2>
The Mother (How I Met Your Mother)
Jump up ^ `` Pilot '' . How I Met Your Mother . Season 1 . Episode 1 . September 19 , 2005 . CBS . ^ Jump up to : Kuchman , Bill ( April 4 , 2014 ) . `` ' How I
Met Your Mother ' ' Last Forever ' Recap : A Divisive Finale and What It Means for ' HIMYM ' '' . Retrieved July 7 , 2014 . Jump up ^ Rullo , Samantha ( 10
March 2014 ) . `` What is The Mother 's Name on ' How I Met Your Mother ' ? Episode Title Might Hold the Secret '' . Retrieved July 7 , 2014 . Jump up ^ ``
HIMYM Tomatometer Watch : Series Finale Currently Rotten - Majority of critics disappointed by ' Last Forever ' '' . Rotten Tomatoes. April 2 , 2014 . Retrieved
April 3 , 2014 .
</item 2>

<item 3>
The Mother (How I Met Your Mother)
In the season 8 episode `` Band or DJ ? , '' Ted runs into Cindy on the subway and tells her that the band Barney and Robin hired to play at their wedding
cancelled at the last minute . The end result of the encounter is that Cindy 's ( now ex - ) roommate 's band plays at Barney and Robin 's wedding . The Mother
is first shown meeting Louis in `` How Your Mother Met Me '' as she is left to carry the band equipment while the now - lead band member Darren talks to his
fans . Later at MacLaren 's Pub , she tells him she 's not yet ready to date . Louis asks her to give him a call if she changes her mind and they begin dating
not long after . The Mother meets all of Ted 's best friends ( Barney , Lily , Marshall and Robin ) before she meets him . The Mother is responsible for
convincing Barney to pursue Robin , as revealed through a flashback in `` Platonish '' . In `` The Locket '' , Tracy meets Lily on a train journey .
</item 3>

<item 4>
The Mother (How I Met Your Mother)
Rolling Stone . Retrieved November 16 , 2014 . ( hide ) How I Met Your Mother Episodes Season 1 Season 2 Season 3 Season 4 Season 5 Season 6 Season 7 Season 8
Season 9 Characters Ted Mosby Marshall Eriksen Lily Aldrin Barney Stinson Robin Scherbatsky The Mother ( Tracy McConnell ) Soundtracks How I Met Your Music
Related articles Awards and nominations The Bro Code Cheerleader effect `` Nothing Suits Me Like a Suit '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=The_Mother_(How_I_Met_Your_Mother)&oldid=802354471 '' Categories : How I Met Your Mother characters Fictional
characters introduced in 2013 Fictional female musicians Hidden categories : Pages using deprecated image syntax Pages using infobox character with unknown
parameters Talk Contents About Wikipedia Español Italiano Magyar Tiếng Việt Edit links This page was last edited on 25 September 2017 , at 17 : 03 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 4>

<item 5>
The Mother (How I Met Your Mother)
Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New
'' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti . The story of how Ted met The Mother is the framing device behind
the series ; many facts about her are revealed throughout the series , including the fact that Ted once unwittingly owned her umbrella before accidentally
leaving it behind in her apartment . Ted and The Mother meet at the Farhampton train station following Barney Stinson and Robin Scherbatsky 's wedding ; this
scene is shown in `` Last Forever '' , the series finale . The Mother 's death from an unspecified terminal illness in 2024 , also revealed in the series finale
, received a mixed reaction from fans . An alternate ending was released in the ninth season DVD . In the alternate ending , Tracy Mosby is still living when
Ted is telling the story in 2030 .
</item 5>

<item 6>
The Mother (How I Met Your Mother)
The Mother ( How I Met Your Mother ) - wikipedia The Mother ( How I Met Your Mother ) Jump to : navigation , search Tracy McConnell How I Met Your Mother
character The Mother appearing in `` The Locket '' First appearance `` Lucky Penny ( unseen ) '' `` Something New '' ( seen ) Last appearance `` Last Forever ''
Created by Carter Bays Craig Thomas Portrayed by Cristin Milioti Information Aliases The Mother Gender Female Spouse ( s ) Ted Mosby Significant other ( s ) Max
( deceased former boyfriend ) Louis ( ex-boyfriend ) Children Penny Mosby ( daughter , born in 2015 , played by Lyndsy Fonseca ) Luke Mosby ( son , born in 2017
, played by David Henrie ) Nationality American Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I
Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother .
</item 6>

<item 7>
The Mother (How I Met Your Mother)
In a flashforward in `` The Lighthouse '' , Ted proposes to the Mother at the top of the lighthouse near Farhampton Inn . She immediately accepts . In another
flashfoward in `` Unpause '' , the Mother is revealed to be pregnant with their second child , Luke , in the year 2017 . She goes into labor while she and Ted
are staying at Farhampton . Name ( edit ) The Mother 's real name is not revealed until the series finale , `` Last Forever '' . When Ted meets her at the
Farhampton train station , she reveals that her name is Tracy McConnell . In the season 1 episode `` Belly Full of Turkey '' , Ted meets a stripper named Tracy
and says `` ... that , kids , is the true story of how I met your mother '' . The children react in surprise and appear to believe Ted before he admits he is
joking , which led some fans to correctly guess that The Mother 's name is Tracy .
</item 7>

<item 8>
The Mother (How I Met Your Mother)
Using Anne Hathaway and Amy Adams as examples , Thomas said that `` We did n't want it to be a big famous star because we did n't want the wider audience to
have associations with whatever actress this would be ... The whole idea is that Ted 's never seen this woman before , so it better feel that way to the
audience '' , similar to how Cobie Smulders being cast as Robin Scherbatsky had `` kept the show alive '' when it began . Bays and Thomas also did not want a
large casting call . They chose Cristin Milioti after seeing her on 30 Rock and Once ; her musical ability was also helpful , as The Mother had been described
as a band member . After competing for the role against at least two others , Milioti filmed her first scene -- for the last episode of season 8 -- having never
watched How I Met Your Mother ; she recalled , `` I had ignorance on my side . So I did n't know what it meant . '' Milioti learned of the character 's
importance only after binge watching the show during the summer .
</item 8>

<item 9>
The Mother (How I Met Your Mother)
After they start talking , Mitch encourages her to pursue her dreams . The Mother expresses her desire to end poverty by taking up economics in college . On his
first day of teaching as Professor Mosby , as seen in the season 4 finale `` The Leap '' , he is seen in front of the classroom of students , one of which
Future Ted says is the titular mother . But in the first episode of season 5 , `` Definitions '' , it is revealed that he was actually in the wrong classroom --
Economics instead of Architecture . At the same time in `` How Your Mother Met Me '' , the Mother sits her first session in Economics 305 and meets another
graduate student named Cindy ( Rachel Bilson ) , whom she offers to move in with her as her roommate . They see Ted enter the room , but when he announces the
subject , The Mother thinks she is in the wrong room and runs off . She heads back to the room after seeing Ted scramble to his actual classroom . Later , in ``
Girls Versus Suits '' , Ted dates Cindy , not knowing that her roommate is his future wife .
</item 9>

<item 10>
The Mother (How I Met Your Mother)
In the video , future Ted is heard saying , `` ... When I think how lucky I am to wake up next to your mom every morning , I ca n't help but be amazed how easy
it all really was ... '' , indirectly stating that The Mother is alive . The video ends right after the train passes at Farhampton station and credits start
rolling , implying that Ted never went back to Robin as he lived a successful married life with Tracy Mosby . Contents ( hide ) 1 Casting 2 Character history
2.1 Name 3 Death 4 Notes 5 References Casting ( edit ) During its first eight seasons , the successful sitcom How I Met Your Mother often hinted at the unseen
character of The Mother . Well - known actresses often made guest appearances on the show . Many fans expected that another would play one of the most - wanted
roles in Hollywood , but creators Carter Bays and Craig Thomas wanted an unknown .
</item 10>
=== END ITEM id=0b725f4689002f20 ===

=== ITEM id=024010219aac1ed9 ===
Gold sentences and facts:
Sentence 0: Year City State , province , dept. , etc . Country Notes 7500 BC Tlapacoya Ixtapaluca Mexico 4000 BC Puerto Hormiga Culture Cartagena Colombia 3710 BC Aspero Norte Chico Peru 2627 BC Caral Norte Chico Peru 1500 BC Kaminaljuyu Guatemala city Guatemala 800 BC Cuicuilco Mexico City Mexico 700 BC Ticul Yucatán Mexico 500 BC Cholula Puebla Mexico 300 BC Teotihuacan México Mexico In the Valley of Mexico 200 Mitla Oaxaca Mexico 200 Acanceh Yucatan Mexico 500 Cuenca Azuay Ecuador 524 Ejutla Oaxaca Mexico ( 1 ) 550 Izamal Yucatan Mexico 600 Cahokia Illinois United States 713 Tula Hidalgo Mexico 1001 L'Anse aux Meadows Newfoundland and Labrador Canada Oldest known European ( Norse ) settlement in the Americas . Possibly founded by Leif Erikson , as part of Vinland , in 1001 AD . 1000 Acoma Pueblo and Taos Pueblo New Mexico United States 1050 Motul Yucatán Mexico 1050 Pachuca Hidalgo Mexico 1100 Cusco Cusco Province Peru 1100 Oraibi Arizona United States ( Hopi Reservation ) ( 2 ) 1168 Ecatepec Estado de Mexico Mexico 1259 Chimaulhuacan Mexico State Mexico 1313 Xalapa Veracruz Mexico 1325 Tenochtitlan Mexico City Mexico Largest pre-Columbian city in the Americas , later called Mexico City . 1450 Zuni Pueblo New Mexico United States ( 3 ) 1470 Iximche Chimaltenango Guatemala 1492 La Navidad Nord - Est Haiti First European settlement in the New World during the Age of Discovery . 1493 La Isabela Puerto Plata Dominican Republic 1498 Santo Domingo Distrito Nacional Dominican Republic Capital of the Dominican Republic . Oldest continuously inhabited European established settlement in the Americas . Founded in 1498 , by Bartholomew Columbus . 1502 Santa Cruz del Seibo El Seibo Dominican Republic 1504 Azua de Compostela Azua Dominican Republic 1504 Santiago de los Caballeros Santiago Dominican Republic 1505 Cotuí Sánchez Ramírez Dominican Republic 1506 Salvaleón de Higüey La Altagracia Dominican Republic 1508 Bonao Monseñor Nouel Dominican Republic 1508 Caparra Puerto Rico United States First city founded by Europeans , although not continuously inhabited , in the United States . Abandoned in 1521 with the removal of the capital to San Juan . 1510 Santa Maria la Antigua del Darien Urabá Colombia First city founded by Europeans on the continent of South America . 1510 Nombre de Dios Colon Panama Oldest continuously inhabited European established settlement in Panama . 1511 Baracoa Guantánamo Cuba Oldest continuously inhabited European established settlement in Cuba , former capital of Cuba . 1513 Bayamo Granma Cuba 1514 Sancti Spiritus Santi Spiritus Cuba 1514 Trinidad Santi Spiritus Cuba 1515 Santiago Santiago After Baracoa and before Havana , Santiago was the second capital of Cuba . Cuba 1519 Havana Havana Cuba 1519 La Villa Rica de la Vera Cruz , ( Veracruz ) Veracruz Mexico Oldest continuously inhabited European established settlement in Mexico . 1519 Panama City Panamá Panama First European established city on the Pacific Coast of the Americas . Founded in 1519 , at the present day ruins of Panama Viejo , it was sacked by the Welsh pirate Henry Morgan in 1671 , and resettled to nearby Casco Viejo , in 1673 . 1520 Hato Mayor del Rey Hato Mayor Dominican Republic 1515 Cumaná Sucre Venezuela Oldest continuously inhabited European established settlement in Venezuela . 1521 San Juan Puerto Rico United States Oldest continuously inhabited European established settlement in Puerto Rico , and in the United States . Oldest continuously inhabited state or territorial capital in the United States . 1524 Granada Granada Nicaragua Oldest continuously inhabited European established settlement in Nicaragua . 1524 Tecpán Guatemala Chimaltenango Guatemala First capital of Guatemala and oldest continuously inhabited European established settlement in Guatemala . 1524 Trujillo Colón Honduras Oldest continuously inhabited European established settlement in Honduras . 1525 Santa Marta Magdalena Colombia Oldest continuously inhabited European established settlement in Colombia . 1525 Acapulco Guerrero Mexico ( 4 ) 1525 San Salvador San Salvador El Salvador Founded in 1525 , rebuilt and changed locations twice afterwards ( 1528 and 1545 ) and oldest continuously inhabited European established settlement in El Salvador . 1526 Acámbaro Guanajuato Mexico 1526 Chimaltenango Chimaltenango Guatemala 1528 San Cristóbal de las Casas Chiapas Mexico Originally called Villa Real de Chiapa . 1528 Camagüey Camagüey Province Cuba Founded in 1515 the northern coast of Camagüey as `` Puerto del Príncipe '' , but moved in 1528 to present location . 1529 Maracaibo Zulia Venezuela 1530 San Blas Nayarit Mexico By Nuño Beltrán ( 5 ) 1531 Puebla Puebla Mexico by Fr . Toribio de Benavente ( `` Motolinía '' ) on 16 April ( 6 ) 1532 Oaxaca Oaxaca Mexico 1532 Piura Piura Peru 1532 São Vicente São Paulo Brazil Oldest continuously inhabited European established settlement in Brazil , oldest Portuguese established settlement in the New World . 1532 Tepic Nayarit Mexico As capital of Nueva Galicia ( 7 ) 1533 Cananéia São Paulo Brazil Second Oldest City in Brazil 1533 Cartagena de Indias Bolívar Colombia 1534 Otavalo Imbabura Ecuador 1534 Quito Pichincha Ecuador Oldest continuously inhabited European established settlement in Ecuador . 1534 Spanish Town Saint Catherine Jamaica Founded by the Spanish as Santiago de la Vega or St. Jago de la Vega . Oldest continuously inhabited European established settlement in Jamaica . 1534 Trujillo La Libertad Peru Oldest continuously inhabited European established settlement in Peru . 1535 Lima Lima Peru 1535 Igarassu Pernambuco Brazil Site of first European settlement in Brazil , the feitoria of Igarassu , in 1516 . 1535 Olinda Pernambuco Brazil One of the best - preserved colonial cities in Brazil . 1535 Vila Velha Espírito Santo Brazil 1535 Paria Oruro Bolivia First Spanish settlement in Bolivia 1536 Santiago de Cali Valle del Cauca Colombia 1536 Popayán Cauca Colombia 1536 San Pedro Sula Cortés Honduras 1536 Tupiza Potosí Bolivia 1537 Asunción Paraguay Oldest continuously inhabited European established settlement in Paraguay . 1537 Recife Pernambuco Brazil Capital of the state of Pernambuco . It is the oldest city among Brazil 's current state capitals . 1538 Bogotá Cundinamarca Colombia 1538 Guayaquil Guayas Ecuador 1538 Iguape São Paulo Brazil 1538 Sucre Chuquisaca Bolivia Originally called Ciudad de la Plata de la Nueva Toledo and the oldest continuously inhabited European established settlement in Bolivia . 1539 Tunja Boyacá Colombia 1539 Vélez Santander Colombia 1539 Pasto Nariño Colombia 1540 Arequipa Arequipa Peru 1540 Ayacucho Ayacucho Peru 1540 Campeche Campeche Mexico 1540 Yuriria Guanajuato Mexico 1540 Mompox Bolivar Colombia 1540 Supía Caldas Colombia 1540 Campeche Campeche Mexico Founded on the site of Mayan village Kimpech . 1541 Huánuco Huánuco Peru 1541 Moquegua Moquegua Peru 1541 Santiago Santiago Chile Oldest continuously inhabited European established settlement in Chile . 1541 Santafé de Antioquia Antioquia Colombia 1542 Guadalajara Jalisco Mexico 1542 Genaro Codina Zacatecas Mexico 1542 San Miguel de Allende Guanajuato Mexico 1542 Mérida Yucatan Mexico Built on the site of Mayan city T'ho . 1542 San Luis de la Paz Guanajuato Mexico 1543 Antigua Guatemala Sacatepéquez Guatemala 1543 Cobán Alta Verapaz Guatemala 1543 Santa Cruz Verapaz Alta Verapaz Guatemala 1545 Potosí Potosí Bolivia 1545 San Juan de los Remedios Villa Clara Province Cuba Remedios has the most uncertain foundation date in Cuba , 1545 has been accepted because it is when the village erected its town hall , but its origin has been studied and documented back to 1513 . 1545 Riohacha La Guajira Colombia 1546 Santos São Paulo Brazil 1546 Neyba Bahoruco Dominican Republic 1546 Tepezalá Aguascalientes Mexico 1548 Asientos Aguascalientes Mexico 1548 La Paz La Paz Bolivia 1548 Pánuco Zacatecas Mexico 1548 Zacatecas Zacatecas Mexico 1549 Salvador Bahia Brazil 1550 Concepción Concepción Chile 1550 Valledupar Cesar Colombia 1550 Ibagué Tolima Colombia 1551 Vitória Espírito Santo Brazil 1551 Mocoa Putumayo 1552 Sonsonate Sonsonate El Salvador 1553 Santiago del Estero Santiago del Estero Argentina Oldest continuously inhabited European established settlement in Argentina . 1553 São Bernardo do Campo São Paulo Brazil 1554 Embu São Paulo Brazil 1554 São Paulo São Paulo Brazil 1555 Saín Alto Zacatecas Mexico 1556 Chalchihuites Zacatecas Mexico 1557 Cuenca ( Santa Ana de los Cuatro Rios de Cuenca ) Azuay Ecuador 1557 Santo Amaro Bahia Brazil 1558 Mérida Mérida Venezuela 1558 Londres Catamarca Argentina Second oldest continuously inhabited European established settlement in Argentina , though refunded and relocated several times . 1560 Mogi das Cruzes São Paulo Brazil 1560 Ipojuca Pernambuco Brazil 1561 Santa Cruz de la Sierra Santa Cruz Bolivia 1561 Mendoza Mendoza Argentina 1561 San Cristobal Táchira Venezuela 1561 Itanhaém São Paulo Brasil 1562 San Juan San Juan Argentina 1562 Río Grande Zacatecas Mexico 1562 Susticacán Zacatecas Mexico 1563 Cartago Cartago Costa Rica Oldest continuously inhabited European established settlement in Costa Rica . 1564 São Mateus Espírito Santo Brazil 1565 Rio de Janeiro Rio de Janeiro Brazil 1565 Saint Augustine Florida United States Oldest continuously inhabited European established settlement in the continental United States . Preceded by Puerto Rican colonies , Pensacola , Florida , which was destroyed in 1559 , and Fort Caroline , destroyed in 1565 . 1565 Tucumán Tucumán Argentina 1567 Caracas Capital District Venezuela 1568 Mazapil Zacatecas Mexico 1568 Goiana Pernambuco Brazil Declared vila in 1685 ; cidade in 1840 . 1568 Valaparaíso Zacatecas Mexico 1569 Santa Ana Santa Ana El Salvador Built over Mayan city of Sihuatehuacán . 1570 Jerez Zacatecas Mexico 1570 Sombrerete Zacatecas Mexico 1571 Cochabamba Cochabamba Bolivia 1572 Huancavelica Huancavelica Peru 1573 San German Puerto Rico United States Second oldest European settlement in Puerto Rico 1573 Córdoba Córdoba Argentina 1573 Santa Fé Santa Fé Argentina 1573 Buga Valle del Cauca Colombia 1574 Huaraz Ancash Peru 1574 Tarija Tarija Department Bolivia 1575 Aguascalientes Aguascalientes Mexico 1576 León Guanajuato Mexico 1577 Juayúa Sonsonate El Salvador 1578 Fort - Liberté Nord - Est Haiti Originally called Bayaha by the Spanish . Oldest continuously inhabited European established settlement in Haiti . 1578 Tegucigalpa Francisco Morazán Honduras 1579 Coamo Puerto Rico United States Third oldest European settlement in Puerto Rico . In the same area where the Taínos had had their village of Guayama . Coamo became officially a town in 1616 , and given the title of villa by Spanish Royal Decree in 1778 . 1580 Buenos Aires Argentina 1580 Santana de Parnaíba São Paulo Brasil 1580 Santo Domingo Xenacoj Sacatepéquez Guatemala 1582 Salta Salta Argentina 1583 St John 's Newfoundland and Labrador Canada Oldest continuously inhabited British settlement in the Americas . 1584 Roanoke North Carolina United States 1585 João Pessoa Paraíba Brazil 1587 Concepción del Oro Zacatecas Mexico 1588 Corrientes Corrientes Argentina 1590 São Cristóvão Sergipe Brazil First capital of Sergipe . Formerly known as Sergipe d'El Rei 1591 Jiménez del Teul Zacatecas Mexico 1591 Juan Aldama Zacatecas Mexico 1591 Paudalho Pernambuco Brazil 1591 La Rioja La Rioja Argentina 1593 San Salvador de Jujuy Jujuy Argentina 1594 Pinos Zacatecas Mexico 1594 San Luis San Luis Argentina 1596 Monterrey Nuevo León Mexico 1596 Villahermosa Tabasco Mexico Originally called Villa Felipe II . 1597 Portobelo Colón Panama 1598 Parras Coahuila Mexico 1599 Natal Rio Grande do Norte Brazil 1599 Tadoussac Quebec Canada Oldest continuously inhabited French established settlement in the Americas , oldest European established settlement in Quebec . 1600 Facatativá Cundinamarca Colombia 1600 Zipaquirá Cundinamarca Colombia 1602 David Chiriquí Province Panama 1603 Salamanca Guanajuato Mexico 1604 Port Royal Nova Scotia Canada Oldest continuously inhabited European established settlement in Nova Scotia . 1606 Bayaguana Monte Plata Dominican Republic 1606 Ibarra Imbabura Ecuador 1606 Oruro Oruro Bolivia 1607 Jamestown Virginia United States First permanent English established settlement in the Americas . 1608 Quebec City Quebec Canada 1609 Fortaleza Ceará Brazil 1610 Kecoughtan Virginia United States 1524 León León Nicaragua Original location ( now known as the ruins of Leon Viejo ) on the south - western shore of Lago Managua founded in 1524 , but destroyed by earthquakes , and resettled 20 miles west , in 1610 , to its present - day location at Leon . 1610 Santa Fe New Mexico United States Oldest continuously inhabited state or territorial capital in the continental United States . ( 8 ) 1610 Harbour Grace Newfoundland and Labrador Canada 1611 Marechal Deodoro , Alagoas Alagoas Brazil 1612 São Luís Maranhão Brazil Capital of Maranhão . 1612 St. George 's Bermuda Oldest continuously inhabited English established settlement in the Americas . 1612 Neiva Huila Colombia 1613 Penedo Alagoas Brazil 1613 Hopewell Virginia United States Oldest continuously inhabited English settlement in North America . Founded in 1613 as Bermuda City by Thomas Dale . 1614 Albany , New York New York United States Oldest US settlement north of Virginia and second oldest state or territorial capital in the continental United States , incorporated 1686 1614 Sirinhaém Pernambuco Brazil Declared vila in 1627 ; cidade in 1892 . 1615 Cabo Frio Rio de Janeiro Brazil 1615 Taos New Mexico United States 1615 Ipiales Nariño Colombia 1616 Belém Pará Brazil Capital of Pará . 1616 Medellín Antioquia Colombia Founded by Francisco de Herrera y Campuzano 1616 Arecibo Puerto Rico United States 1617 Jersey City , New Jersey New Jersey United States Pavonia , New Netherland 1618 Cabo de Santo Agostinho Pernambuco Brazil Declared vila in 1811 ; cidade in 1877 . 1620 Ojocaliente Zacatecas Mexico 1620 Plymouth Massachusetts United States 1622 Bucaramanga Santander Colombia 1623 Dover New Hampshire United States 1623 Gloucester Massachusetts United States 1624 Old Road Saint Christopher Saint Kitts and Nevis Oldest continuously inhabited British established settlement in the Caribbean , First successful ' non-Spanish ' established settlement in the Caribbean . 1624 Villa Soriano Soriano Uruguay Oldest continuously inhabited European established settlement in Uruguay . 1625 New Amsterdam New York United States Now New York City . 1625 Quincy Massachusetts United States 1626 Salem Massachusetts United States 1626 Vitória de Santo Antão Pernambuco Brazil Declared vila in 1811 ; cidade in 1843 . 1627 Basseterre Saint Christopher Saint Kitts and Nevis Oldest continuously inhabited French established settlement in the Caribbean . 1628 Bridgetown Saint Michael Barbados Second oldest continuously inhabited English established settlement in the Caribbean . 1630 Boston Massachusetts United States 1630 Paramaribo Paramaribo Suriname Oldest continuously inhabited European established settlement in Suriname . 1630 Speightstown Saint Peter Barbados 1631 Biddeford Maine United States 1631 Hidalgo del Parral Chihuahua Mexico 1631 Saint John New Brunswick Canada 1632 Batopilas Chihuahua Mexico 1632 Williamsburg Virginia United States 1632 St. John 's Antigua Antigua and Barbuda 1634 Green Bay Wisconsin United States Oldest continuously inhabited French established settlement in the United States . 1634 St. Mary 's City Maryland United States 1634 Trois - Rivières Quebec Canada 1634 Willemstad Curaçao Netherlands Oldest continuously inhabited Dutch established settlement in the Caribbean . 1635 Concord Massachusetts United States 1635 Old Saybrook Connecticut United States Original Dutch settlement called Kievits Hoek . 1636 Springfield Massachusetts United States 1636 Providence Rhode Island United States 1637 Hartford Connecticut United States ( 9 ) 1638 Belize City Belize Belize Oldest continuously inhabited European established settlement in Belize . Original Mayan city called Holzuz . 1638 Fort - de-France Martinique Originally called Fort Saint Louis . 1638 New Haven Connecticut United States ( 10 ) 1638 Wilmington Delaware United States Grew from Fort Christina , part of the New Sweden colony . Oldest continuously inhabited Swedish established settlement in the Americas . 1638 Cambridge Massachusetts United States 1639 Rincón de Romos Aguascalientes Mexico 1639 ( prior to ) St. Marks Florida United States ( 11 ) 1639 Barnstable Massachusetts United States Second - oldest municipality and only city on Cape Cod 1639 Bridgeport Connecticut United States 1639 Guilford Connecticut United States Location of the Henry Whitfield House , the oldest house in Connecticut and the oldest stone house in New England 1639 Milford Connecticut United States 1639 Newport Rhode Island United States 1640 Tumaco Nariño Colombia 1642 Montreal Quebec Canada 1642 Lexington Massachusetts United States ( 12 ) 1642 Warwick Rhode Island United States 1643 Basse - Terre Guadeloupe France Territorial capital . 1643 Dolores Hidalgo Guanajuato Mexico 1644 Salvatierra Guanajuato Mexico 1646 New London Connecticut United States 1648 Alcântara Maranhão Brazil 1648 Paranaguá Paraná Brazil 1649 Annapolis Maryland United States 1650 Castries Castries Saint Lucia 1650 Saint George 's Saint George Grenada 1651 Norwalk Connecticut United States 1654 Vieux Fort Vieux Fort Saint Lucia 1659 Ciudad Juárez Chihuahua Mexico 1659 El Paso Texas United States 1660 Charlestown Nevis St Kitts & Nevis 1660 Placentia Newfoundland and Labrador Canada French Capital until 1713 , originally called Plaisance 1660 Rye New York United States 1664 Cayenne French Guiana France Oldest continuously inhabited European established settlement in French Guiana . 1664 Quibdó Chocó Colombia 1665 Port - de-Paix Nord - Ouest Haiti 1666 Codrington Barbuda Antigua and Barbuda 1666 Newark New Jersey United States 1667 Paraty Rio de Janeiro Brazil 1668 Sault Ste . Marie Michigan United States ( 13 ) , oldest city in Michigan . 1670 Boqueirão Paraíba Brazil 1670 Cap - Haïtien Nord Haiti Originally called `` Cap - Français '' 1670 Charleston South Carolina United States 1672 Charlotte Amalie , St. Thomas U.S. Virgin Islands United States Oldest permanent European settlement in Saint Thomas island . Oldest continuously inhabited Danish established settlement in the Americas . 1673 Kingston Ontario Canada Grew from Fort Frontenac . Continuously inhabited since 1784 . 1673 San José de Gracia Aguascalientes Mexico 1673 Worcester Massachusetts United States 1674 Cachoeira Bahia Brazil 1674 Waterbury Connecticut United States 1677 Crosswicks New Jersey United States Quaker settlement . Continuously inhabited since founding . 1680 South Orange New Jersey United States Grew from Newark ( later Orange ) . 1680 Colonia del Sacramento Colonia Uruguay 1681 Cockburn Town Turks and Caicos Islands 1681 Philadelphia Pennsylvania United States 1682 Norfolk Virginia United States 1682 São Borja Rio Grande do Sul Brazil 1683 Dover Delaware United States 1685 Escada Pernambuco Brazil 1686 Trinidad Beni Department Bolivia 1687 New Britain Connecticut United States 1687 São Luiz Gonzaga Rio Grande do Sul Brazil 1687 São Miguel das Missões Rio Grande do Sul Brazil 1687 São Nicolau Rio Grande do Sul Brazil 1689 Marigot Saint Martin 1689 San Gil Santander Colombia 1690 Port of Spain Trinidad Trinidad and Tobago 1690 São Lourenço das Missões Rio Grande do Sul Brazil 1690 Santa Rosa de Viterbo Boyacá Colombia 1692 Villanueva Zacatecas Mexico 1693 Bom Jesus da Lapa Bahia Brazil 1693 Burlington New Jersey United States 1693 Curitiba Paraná Brazil 1693 Kingston St Andrew Jamaica 1695 Nassau New Providence Bahamas 1695 Saint - Marc Artibonite Haiti 1696 Assú Rio Grande do Norte Brazil Declared vila in 1766 ; cidade in 1845 . 1696 Rimouski Quebec Canada 1698 Ambato Tungurahua Ecuador 1698 Pensacola Florida United States Known for being the first European settlement in North America . 1699 Biloxi Mississippi United States 1699 Baton Rouge Louisiana United States 1700 Barreiras Bahia Brazil 1701 Detroit Michigan United States 1701 São João del Rei Minas Gerais Brazil 1702 Jesús María Aguascalientes Mexico 1702 Mobile Alabama United States 1703 Kaskaskia Illinois United States 1705 Bath North Carolina United States 1706 Albuquerque New Mexico United States 1706 Santo Ângelo Rio Grande do Sul Brazil 1709 Chihuahua Chihuahua Mexico 1710 Chatham New Jersey United States On land purchased in 1680 . 1710 New Bern North Carolina United States Birthplace of Pepsi 1711 Beaufort South Carolina United States 1711 Ouro Preto Minas Gerais Brazil 1714 Natchitoches Louisiana United States Oldest settlement in the Louisiana Purchase 1714 Antonina Paraná Brazil 1714 Laguna Santa Catarina Brazil 1714 Serro Minas Gerais Brazil 1715 ( prior to ) Kekionga Indiana United States Capital of the Miami tribe . 1716 Natchez Mississippi United States Dates to the founding of Fort Rosalie by the French. ( 14 ) 1717 Brejo do Cruz Paraíba Brazil 1717 Zacatecas Zacatecas Mexico 1717 Westborough Massachusetts United States One Hundredth Town in Massachusetts 1718 New Orleans Louisiana United States 1718 Rio de Contas Bahia Brazil 1718 San Antonio Texas United States 1718 Tiradentes Minas Gerais Brazil 1719 Trenton New Jersey United States 1721 Cortazar Guanajuato Mexico 1722 Edenton North Carolina United States 1723 Guaynabo Puerto Rico United States Declared pueblo in 1723 ; municipio in 1768 . 1724 Brattleboro Vermont United States Grew out of Fort Dummer . 1725 Concord New Hampshire United States 1726 Florianópolis Santa Catarina Brazil 1726 Montevideo Montevideo Uruguay 1727 Cuiabá Mato Grosso Brazil 1727 Goiás Goiás Brazil 1727 Pirenópolis Goiás Brazil 1728 Fredericksburg Virginia United States 1728 Inhambupe Bahia Brazil 1728 Nuuk Greenland Norse colony was originally called Godthab . Oldest continuously inhabited European established settlement in Greenland . 1729 Baltimore Maryland United States 1730 Roseau Saint George Dominica 1732 Fredericton New Brunswick Canada Originally called Ste . Anne 's Point . 1732 Vincennes Indiana United States 1733 Morretes Paraná Brazil 1733 Pau dos Ferros Rio Grande do Norte Brazil 1733 Richmond Virginia United States ( 15 ) 1733 Savannah Georgia United States 1733 Cúcuta Norte de Santander Colombia 1736 Augusta Georgia United States 1737 Rio Grande Rio Grande do Sul Brazil 1738 Fort Rouge Manitoba Canada Now Winnipeg , Manitoba . 1738 San José San José Costa Rica 1740 Santa Rosalía de Camargo Chihuahua Mexico 1740 Portalegre Rio Grande do Norte Brazil 1740 San Felipe de Puerto Plata Puerto Plata Dominican Republic 1741 Viamão Rio Grande do Sul Brazil 1743 Canguaretama Rio Grande do Norte Brazil Declared vila in 1858 ; cidade in 1885 . 1748 Caicó Rio Grande do Norte Brazil Declared vila in 1788 ; cidade in 1868 . 1749 Alexandria Virginia United States 1749 San Ignacio de Velasco Santa Cruz Department Bolivia 1749 Port - au - Prince Ouest Haiti 1749 Halifax Nova Scotia Canada 1749 Windsor Ontario Canada Oldest continually - inhabited settlement in Canada west of Montreal 1751 Manchester New Hampshire United States Former American textile capital - Named after Manchester , England 1752 Buíque Pernambuco Brazil 1752 Patos Paraíba Brazil Declared vila in 1832 ; cidade in 1903 . 1752 Tubac Arizona United States 1754 Augusta Maine United States 1755 Charlotte North Carolina United States 1756 Salto Salto Uruguay 1756 Santa Bárbara de Samaná Samaná Dominican Republic 1756 Yauco Puerto Rico United States 1757 Estância Sergipe Brazil 1758 Pittsburgh Pennsylvania United States 1758 Macapá Amapá Brazil 1760 Arês Rio Grande do Norte Brazil 1760 Sabana de la Mar Hato Mayor Dominican Republic 1761 Sacaba Cochabamba Department Bolivia 1762 Kingstown St Vincent Saint Vincent and the Grenadines 1762 Parnaíba Piauí Brazil 1762 Shepherdstown West Virginia United States Originally known as Mecklenburg . 1762 Allentown Pennsylvania United States Incorporated as Northamptontown . 1762 Baía da Traição Paraíba Brazil 1763 Burlington Vermont United States 1763 Pánfilo Natera Zacatecas Mexico 1763 Ellsworth Maine United States 1763 Philipsburg Sint Maarten Netherlands 1763 St. Louis Missouri United States 1764 Charlottetown Prince Edward Island Canada 1764 Pointe - à - Pitre Grand Terre Guadeloupe 1765 Port Elizabeth Grenadines Saint Vincent and the Grenadines 1765 Portsmouth Saint John Dominica 1767 Ceará - Mirim Rio Grande do Norte Brazil 1767 Araioses Maranhão Brazil 1769 Elizabethton Tennessee United States 1769 Falmouth Trelawny Jamaica 1769 San Diego California United States Grew from Presidio of San Diego 1769 Lapa Paraná Brazil 1770 Monterey California United States Grew from Presidio of Monterey . Original capital of California ( 16 ) 1770 Araxá Minas Gerais Brazil 1771 Calvillo Aguascalientes Mexico 1772 Porto Alegre Rio Grande do Sul Brazil 1775 Tucson Arizona United States 1774 Castro Paraná Brazil 1775 Lexington Kentucky United States 1775 Boonesborough Kentucky United States Grew from Fort Boonesborough , built by pioneer Daniel Boone . - 1775 São Lourenço da Mata Pernambuco Brazil District status in 1775 ; village status in 1884 ; municipality status in 1890 1776 San Francisco California United States 1776 Sincelejo Sucre Colombia 1777 San Jose California United States Originally known as El Pueblo de San José de Guadalupe , the first town in the Spanish colony of Nueva California , which later became Alta California . 1777 Montería Cordoba Colombia 1778 Corumbá Mato Grosso do Sul Brazil 1778 Louisville Kentucky United States Grew from Fort Nelson , established by explorer George Rogers Clark . 1778 San Francisco de Macorís Duarte Dominican Republic 1779 Jonesborough Tennessee United States Later organized as the lost State of Franklin with Jonesborough as capital 1784 . 1779 Nashville Tennessee United States Grew from Fort Nashborough 1780 Codó Maranhão Brazil Declared vila in 1833 ; cidade in 1896 . 1780 Las Matas de Farfán San Juan Dominican Republic 1780 Arauca Arauca Colombia 1781 Montpelier Vermont United States 1781 Los Angeles California United States 1782 Catu Bahia Brazil 1782 Georgetown Demerara - Mahaica Guyana Originally called La Nouvelle Ville . 1783 Vitória da Conquista Bahia Brazil 1784 San Fernando Trinidad Trinidad and Tobago 1785 Harrisburg Pennsylvania United States 1785 North Battleford Saskatchewan Canada 1785 fur trading post . 1877 Battleford is capital of the North West Territories . North Battleford incorporated as a city ( population 5000 + ) 1913 . 1785 Asheville North Carolina United States 1785 Gustavia Saint Barthélemy 1785 Huntington West Virginia United States 1785 Sydney Nova Scotia Canada 1786 Columbia South Carolina United States 1786 Escada Pernambuco Brazil Declared vila in 1854 ; cidade in 1873 . 1786 Florissant Missouri United States Originally known as St. Ferdinand . 1786 Frankfort Kentucky United States 1786 Portland Maine United States 1787 Boa Ventura Paraíba Brazil 1788 Marietta Ohio United States 1788 Cincinnati Ohio United States 1788 Charleston West Virginia United States Grew from Fort Lee . ( 17 ) 1788 Mercedes Soriano Uruguay 1789 Buffalo New York United States 1789 Itambé Pernambuco Brazil Declared vila in 1867 ; cidade in 1879 . 1790 Hamilton Bermuda 1790 Washington District of Columbia United States 1791 Jacksonville Florida United States Known as `` Cowford '' . 1791 Knoxville Tennessee United States 1791 Bangor Maine United States 1792 Raleigh North Carolina United States 1793 Sherbrooke Quebec Canada 1793 Toronto Ontario Canada 1794 Fort Wayne Indiana United States 1795 Edmonton Alberta Canada Grew from Fort Edmonton . 1795 Erie Pennsylvania United States Grew from the French Fort Presque Isle . 1795 Maryville Tennessee United States Grew from the American Fort Craig . 1796 Cleveland Ohio United States 1796 Oranjestad Aruba Netherlands 1797 Chaguanas Trinidad Trinidad and Tobago 1797 Franklinton Ohio United States Eventually absorbed by Columbus , Ohio . 1799 Araruama Rio de Janeiro Brazil 1800 Hull Quebec Canada Formerly known as Wrightville . 1800 Sonsón Antioquia Colombia 1802 Chapadinha Maranhão Brazil 1802 Santa Cruz de Barahona Barahona Dominican Republic 1803 Chicago Illinois United States Grew from Fort Dearborn . 1805 Huntsville Alabama United States 1807 Hot Springs Arkansas United States 1809 Água Preta Pernambuco Brazil Declared vila in 1846 ; cidade in 1895 . 1810 Guarapuava Paraná Brazil 1810 Manchester New Hampshire United States 1810 Pilão Arcado Bahia Brazil 1810 San Bernardino California United States 1811 Garanhuns Pernambuco Brazil 1811 Astoria Oregon United States 1812 Columbus Ohio United States 1812 Pelotas Rio Grande do Sul Brazil Declared vila in 1832 ; cidade in 1835 . 1813 Capela Sergipe Brazil 1813 Chisec Alta Verapaz Guatemala 1813 Barranquilla Atlántico Colombia 1815 Areia Paraíba Brazil 1815 Hamilton Ontario Canada 1816 Alagoinhas Bahia Brazil 1816 Chattanooga Tennessee United States Originally named Ross 's Landing . 1816 Saginaw Michigan United States 1816 Savaneta Aruba Netherlands 1817 Fort Smith Arkansas United States 1818 Itaguaí Rio de Janeiro Brazil 1819 Montgomery Alabama United States Near the site of the earlier French Fort Toulouse . 1819 Springfield Illinois United States 1819 Managua Managua Nicaragua 1819 Memphis Tennessee United States Near the site of the earlier French Fort Prudhomme . 1819 Palmeira Paraná Brazil 1819 Tuscaloosa Alabama United States Former state capital , grew from former Native American settlements beginning in 1580 . ( 18 ) 1820 Cotegipe Bahia Brazil 1821 Little Rock Arkansas United States 1821 Indianapolis Indiana United States 1821 Jefferson City Missouri United States 1821 Decatur Alabama United States 1822 Ponta Grossa Paraná Brazil 1822 Jackson Mississippi United States 1823 Tampa Florida United States Grew from earlier military post Fort Brooke . 1824 Vancouver Washington United States 1824 Tallahassee Florida United States 1825 Vicksburg Mississippi United States 1825 Grand Rapids Michigan United States 1825 Syracuse New York United States incorporated as village ; received city charter in 1847 1826 Ipiranga Paraná Brazil 1826 London Ontario Canada 1826 Bytown Ontario Canada Now Ottawa , Ontario . 1827 Fort Leavenworth Kansas United States 1827 St. Andrews Florida United States Now part of Panama City 1831 Greenville South Carolina United States 1831 Piracanjuba Goiás Brazil 1832 Goianinha Rio Grande do Norte Brazil 1828 Key West Florida United States 1832 Touros Rio Grande do Norte Brazil 1833 Bananeiras Paraíba Brazil 1833 Juazeiro Bahia Brazil 1833 Kitchener Ontario Canada Originally called Berlin . 1833 Milwaukee Wisconsin United States 1833 Nazaré da Mata Pernambuco Brazil Declared vila in 1833 ; cidade in 1850 . 1833 Rio Formoso Pernambuco Brazil Declared vila in 1833 ; cidade in 1850 . 1833 Vassouras Rio de Janeiro Brazil 1835 Austin Texas United States 1835 Barra do Corda Maranhão Brazil 1835 Curupuru Maranhão Brazil Declared vila in 1841 ; cidade in 1920 . 1836 Davenport Iowa United States 1836 Oshawa Ontario Canada 1836 Shreveport Louisiana United States 1836 Madison Wisconsin United States 1836 Tulsa Oklahoma United States The Lochapoka named their new settlement `` Tulasi , '' meaning `` old town '' 1837 Glória do Goitá Pernambuco Brazil Declared vila in 1877 ; cidade in 1884 . 1837 Lansing Michigan United States 1837 Houston Texas United States 1838 Divina Pastora Sergipe Brazil 1838 Kansas City Missouri United States 1838 St Paul Minnesota United States 1839 Santa Rita Paraíba Brazil Declared vila in 1890 ; cidade in 1924 . 1840 Summerside Prince Edward Island Canada 1840 Villavicencio Meta Colombia 1841 Cedar Rapids Iowa United States 1841 Dallas Texas United States 1842 Piatã Bahia Brazil 1842 Salem Oregon United States 1843 Atlanta Georgia United States Originally known as Terminus , later renamed Marthasville . Acquired the name Atlanta in 1847 . 1843 Des Moines Iowa United States 1843 Stanley East Falkland Falkland Islands 1843 Uruguaiana Rio Grande do Sul Brazil Declared vila in 1846 ; cidade in 1874 . 1843 Victoria British Columbia Canada 1844 Lençóis Bahia Brazil 1845 Araruna Paraíba Brazil 1845 Portland Oregon United States 1847 Alagoa Grande Paraíba Brazil Declared vila in 1865 ; cidade in 1908 . 1847 Salt Lake City Utah United States Originally known as Great Salt Lake City . 1848 Caruaru Pernambuco Brazil First municipality in the ( Agreste ) region of Pernambuco state , Brazil . 1848 Rio Verde Goiás Brazil 1848 Sacramento California United States 1849 Provo Utah United States 1849 Manizales Caldas Colombia 1851 San Luis Colorado United States 1851 Seattle Washington United States 1851 Jericó Antioquia Colombia 1852 Mossoró Rio Grande do Norte Brazil 1852 São José dos Pinhais Paraná Brazil 1852 Colón Colón Panama 1853 Olympia Washington United States 1854 Omaha Nebraska United States 1854 Anajatuba Maranhão Brazil 1854 Topeka Kansas United States 1855 Aracaju Sergipe Brazil 1856 Lincoln Nebraska United States Originally called Lancaster . 1856 O'Fallon Missouri United States 1856 Sioux Falls South Dakota United States 1857 Boquim Sergipe Brazil 1858 Carson City Nevada United States 1858 Denver Colorado United States 1859 Yankton South Dakota United States 1860 Chico California United States 1863 Boise Idaho United States 1863 Wichita Kansas United States 1864 Casper Wyoming United States Originally called Platte Bridge Station . 1864 Helena Montana United States 1864 Salinas California United States ( 19 ) 1866 Prince Albert Saskatchewan Canada Settled by Rev. James Nisbet . Incorporated as city ( population 5000 + ) October 8 , 1904 1867 Cheyenne Wyoming United States 1867 Gameleira Pernambuco Brazil 1867 Minneapolis Minnesota United States 1868 Colinas Maranhão Brazil Declared vila in 1870 ; cidade in 1891 . 1868 Phoenix Arizona United States 1870 Afogados da Ingazeira Pernambuco Brazil 1870 Campo Largo Paraná Brazil 1870 Puerto Limon Limon Costa Rica 1871 Águas Belas Pernambuco Brazil 1871 Alto Parnaíba Maranhão Brazil 1871 Birmingham Alabama United States 1871 Colorado Springs Colorado United States 1871 Fargo North Dakota United States 1871 Imbituva Paraná Brazil 1871 São João do Triunfo Paraná Brazil 1872 Anniston Alabama United States 1872 Bismarck North Dakota United States 1872 Tibagi Paraná Brazil 1873 Jaboatão dos Guararapes Pernambuco Brazil First settlement in 1593 ; village status in 1873 ; city status in 1884 1873 Timbaúba Pernambuco Brazil Declared vila in 1879 ; cidade in 1884 . 1875 Orlando Florida United States 1875 Calgary Alberta Canada Grew from Fort Calgary 1875 Humboldt Saskatchewan Canada Incorporated as a city ( population 5000 + ) November 7 , 2000 1876 Boa Vista Paraíba Brazil 1876 Cajapió Maranhão Brazil 1876 Rapid City South Dakota United States 1877 Billings Montana United States 1877 Jaquimeyes Barahona Dominican Republic 1878 Coeur d'Alene Idaho United States 1878 Colombo Paraná Brazil 1879 Vicência Pernambuco Brazil Declared vila in 1891 ; cidade in 1928 . 1880 Pierre South Dakota United States 1881 Juneau Alaska United States 1881 Moose Jaw Saskatchewan Canada Incorporated as a city ( population 5000 + ) November 1903 . 1882 Brandon Manitoba Canada 1882 Mao Valverde Dominican Republic 1882 Restauración Dajabón Dominican Republic 1882 Yorkton Saskatchewan Canada The York Farmers ' Colonization Company Limited settlement May 12 , 1882 . City ( population 5000 + ) incorporated February 1 , 1928 . 1882 Swift Current Saskatchewan Canada Canadian Pacific Railway arrival December 10 , 1882 . City ( population 5000 + ) incorporated January 15 , 1914 . 1882 Regina Saskatchewan Canada Settlement first named Pile O ' Bones , and renamed Regina . Capital city of SK . Incorporated as a city ( population 5000 + ) June 19 , 1903 1882 Rio Branco Acre Brazil 1883 Saskatoon Saskatchewan Canada Temperance Colony founded at Nutana . Riversdale , Nutana , and Saskatoon incorporated as a city ( population 5000 + ) in 1906 . 1884 Melfort Saskatchewan Canada Stoney Creek Settlement began in 1884 . Incorporated as a city ( population 5000 + ) on July 1 , 1907 . 1886 Vancouver British Columbia Canada 1887 Gulfport Mississippi United States ( 20 ) 1888 Virginia Beach Virginia United States Grew from Seatack community . 1889 Oklahoma City Oklahoma United States 1889 Whiting Indiana United States 1890 Ijuí Rio Grande do Sul Brazil 1892 Aliança Pernambuco Brazil Declared vila in 1909 ; cidade in 1928 . 1892 Catende Pernambuco Brazil Created in 1892 as a district of Palmares . Autonomous municipality in 1909 . 1892 Estevan Saskatchewan Canada CPR arrives 1893 . Incorporated as city ( population 5000 + ) March 1 , 1957 1892 Guayaramerín Beni Bolivia 1892 Restauración Dajabón Dominican Republic 1892 Vernon British Columbia Canada 1893 Araripina Pernambuco Brazil Created in 1893 as a district of Ouricuri . Autonomous municipality in 1928 . 1893 Catuípe Rio Grande do Sul Brazil 1894 Riberalta Beni Bolivia 1895 Traverse City Michigan United States 1896 Condado Pernambuco Brazil Declared cidade in 1958 . 1896 Inhumas Goiás Brazil 1896 Miami Florida United States 1896 Santa Catalina la Tinta Alta Verapaz Guatemala 1897 Jequié Bahia Brazil 1897 Nelson British Columbia Canada 1898 Weyburn Saskatchewan Canada Signal Hill settlement . Weyburn incorporated as a city ( population 5000 + ) 1913 . 1898 Chetumal Quintana Roo Mexico Originally called Payo Obispo . 1899 Porto Acre Acre Brazil 1900 Afrânio Pernambuco Brazil Created in 1900 as a district of Petrolina . Autonomous municipality in 1963 . 1901 Fairbanks Alaska United States 1902 Florencia Caquetá Colombia 1903 Lloydminster Saskatchewan Canada Brittania colony settled by Rev. Exton Lloyd and Rev. Isaac Barr . Incorporated as a city ( population 5000 + ) in 1958 . 1904 Cruzeiro do Sul Acre Brazil 1904 Grytviken South Georgia 1904 Sena Madureira Acre Brazil 1904 Xapuri Acre Brazil 1906 Cobija Pando Bolivia 1906 Feijó Acre Brazil 1907 Melville Saskatchewan Canada Incorporated as a city ( population 5000 + ) August 1 , 1960 1907 Tarauacá Acre Brazil 1908 São Mateus do Sul Paraná Brazil 1909 Estreito Maranhão Brazil 1910 Brasiléia Acre Brazil 1910 Kindersley Saskatchewan Canada 1910 Porto Walter Acre Brazil 1910 Villazón Potosí Bolivia 1911 Las Vegas Nevada United States 1914 Anchorage Alaska United States 1915 Yopal Casanare Colombia 1928 Leticia Amazonas Colombia 1929 Pabellón de Arteaga Aguascalientes Mexico 1930 Puerto Carreño Vichada Colombia 1931 Loreto Zacatecas Mexico 1935 Yellowknife Northwest Territories Canada 1935 Mitú Vaupés Colombia 1938 San José del Guaviare Guaviare Colombia 1942 Iqaluit Nunavut Canada Originally called Frobisher Bay . 1955 Freeport Grand Bahama Bahamas 1957 Ciudad del Este Alto Paraná Paraguay Originally called Puerto Flor de Lis . 1957 Rankin Inlet Nunavut Canada 1960 Brasília Distrito Federal Brazil Created in 1960 as the national capital . 1970 Belmopan Cayo Belize 1970 Linden Upper Demerara - Berbice Guyana City formed by combining the towns of Christianburg , MacKenzie , and Wismar . 1970 Cancún Quintana Roo Mexico Inírida Guainía Colombia Palmas Tocantins Brazil Was founded 1 year after the creation of the State of Tocantins . It 's the newest and smallest Brazilian capital city .
  Fact 0.0: Tlapacoya (Ixtapaluca, Mexico State, Mexico) was founded in 7500 BC.
  Fact 0.1: Puerto Hormiga Culture (Cartagena, Colombia) was founded in 4000 BC.
  Fact 0.2: Aspero (Norte Chico, Peru) was founded in 3710 BC.
  Fact 0.3: Caral (Norte Chico, Peru) was founded in 2627 BC.
  Fact 0.4: Kaminaljuyu (Guatemala city, Guatemala) was founded in 1500 BC.
  Fact 0.5: Cuicuilco (Mexico City, Mexico) was founded in 800 BC.
  Fact 0.6: Ticul (Yucatan, Mexico) was founded in 700 BC.
  Fact 0.7: Cholula (Puebla, Mexico) was founded in 500 BC.
  Fact 0.8: Teotihuacan (Mexico State, Mexico) was founded in 300 BC.
  Fact 0.9: Teotihuacan is in the Valley of Mexico.
  Fact 0.10: Mitla (Oaxaca, Mexico) was founded in 200.
  Fact 0.11: Acanceh (Yucatan, Mexico) was founded in 200.
  Fact 0.12: Cuenca (Azuay, Ecuador) was founded in 500.
  Fact 0.13: Ejutla (Oaxaca, Mexico) was founded in 524.
  Fact 0.14: Izamal (Yucatan, Mexico) was founded in 550.
  Fact 0.15: Cahokia (Illinois, United States) was founded in 600.
  Fact 0.16: Tula (Hidalgo, Mexico) was founded in 713.
  Fact 0.17: L'Anse aux Meadows (Newfoundland and Labrador, Canada) was founded in 1001.
  Fact 0.18: L'Anse aux Meadows is the oldest known European (Norse) settlement in the Americas.
  Fact 0.19: L'Anse aux Meadows was possibly founded by Leif Erikson as part of Vinland in 1001 AD.
  Fact 0.20: Acoma Pueblo and Taos Pueblo (New Mexico, United States) were founded in 1000.
  Fact 0.21: Motul (Yucatan, Mexico) was founded in 1050.
  Fact 0.22: Pachuca (Hidalgo, Mexico) was founded in 1050.
  Fact 0.23: Cusco (Cusco Province, Peru) was founded in 1100.
  Fact 0.24: Oraibi (Arizona, United States, Hopi Reservation) was founded in 1100.
  Fact 0.25: Ecatepec (Estado de Mexico, Mexico) was founded in 1168.
  Fact 0.26: Chimaulhuacan (Mexico State, Mexico) was founded in 1259.
  Fact 0.27: Xalapa (Veracruz, Mexico) was founded in 1313.
  Fact 0.28: Tenochtitlan (Mexico City, Mexico) was founded in 1325.
  Fact 0.29: Tenochtitlan was the largest pre-Columbian city in the Americas.
  Fact 0.30: Tenochtitlan was later called Mexico City.
  Fact 0.31: Zuni Pueblo (New Mexico, United States) was founded in 1450.
  Fact 0.32: Iximche (Chimaltenango, Guatemala) was founded in 1470.
  Fact 0.33: La Navidad (Nord-Est, Haiti) was founded in 1492.
  Fact 0.34: La Navidad was the first European settlement in the New World during the Age of Discovery.
  Fact 0.35: La Isabela (Puerto Plata, Dominican Republic) was founded in 1493.
  Fact 0.36: Santo Domingo (Distrito Nacional, Dominican Republic) was founded in 1498.
  Fact 0.37: Santo Domingo is the capital of the Dominican Republic.
  Fact 0.38: Santo Domingo is the oldest continuously inhabited European established settlement in the Americas.
  Fact 0.39: Santo Domingo was founded in 1498 by Bartholomew Columbus.
  Fact 0.40: Santa Cruz del Seibo (El Seibo, Dominican Republic) was founded in 1502.
  Fact 0.41: Azua de Compostela (Azua, Dominican Republic) was founded in 1504.
  Fact 0.42: Santiago de los Caballeros (Santiago, Dominican Republic) was founded in 1504.
  Fact 0.43: Cotui (Sanchez Ramirez, Dominican Republic) was founded in 1505.
  Fact 0.44: Salvaleon de Higuey (La Altagracia, Dominican Republic) was founded in 1506.
  Fact 0.45: Bonao (Monseñor Nouel, Dominican Republic) was founded in 1508.
  Fact 0.46: Caparra (Puerto Rico, United States) was founded in 1508.
  Fact 0.47: Caparra was the first city founded by Europeans in the United States, although not continuously inhabited.
  Fact 0.48: Caparra was abandoned in 1521 with the removal of the capital to San Juan.
  Fact 0.49: Santa Maria la Antigua del Darien (Uraba, Colombia) was founded in 1510.
  Fact 0.50: Santa Maria la Antigua del Darien was the first city founded by Europeans on the South American continent.
  Fact 0.51: Nombre de Dios (Colon, Panama) was founded in 1510.
  Fact 0.52: Nombre de Dios is the oldest continuously inhabited European established settlement in Panama.
  Fact 0.53: Baracoa (Guantanamo, Cuba) was founded in 1511.
  Fact 0.54: Baracoa is the oldest continuously inhabited European established settlement in Cuba.
  Fact 0.55: Baracoa is a former capital of Cuba.
  Fact 0.56: Bayamo (Granma, Cuba) was founded in 1513.
  Fact 0.57: Sancti Spiritus (Santi Spiritus, Cuba) was founded in 1514.
  Fact 0.58: Trinidad (Santi Spiritus, Cuba) was founded in 1514.
  Fact 0.59: Santiago (Santiago, Cuba) was founded in 1515.
  Fact 0.60: Santiago was the second capital of Cuba, after Baracoa and before Havana.
  Fact 0.61: Havana (Havana, Cuba) was founded in 1519.
  Fact 0.62: La Villa Rica de la Vera Cruz (Veracruz, Mexico) was founded in 1519.
  Fact 0.63: La Villa Rica de la Vera Cruz is the oldest continuously inhabited European established settlement in Mexico.
  Fact 0.64: Panama City (Panama, Panama) was founded in 1519.
  Fact 0.65: Panama City was the first European established city on the Pacific Coast of the Americas.
  Fact 0.66: Panama City was founded at the present day ruins of Panama Viejo.
  Fact 0.67: Panama City was sacked by the Welsh pirate Henry Morgan in 1671.
  Fact 0.68: Panama City was resettled to nearby Casco Viejo in 1673.
  Fact 0.69: Hato Mayor del Rey (Hato Mayor, Dominican Republic) was founded in 1520.
  Fact 0.70: Cumana (Sucre, Venezuela) was founded in 1515.
  Fact 0.71: Cumana is the oldest continuously inhabited European established settlement in Venezuela.
  Fact 0.72: San Juan (Puerto Rico, United States) was founded in 1521.
  Fact 0.73: San Juan is the oldest continuously inhabited European established settlement in Puerto Rico.
  Fact 0.74: San Juan is the oldest continuously inhabited European established settlement in the United States.
  Fact 0.75: San Juan is the oldest continuously inhabited state or territorial capital in the United States.
  Fact 0.76: Granada (Granada, Nicaragua) was founded in 1524.
  Fact 0.77: Granada is the oldest continuously inhabited European established settlement in Nicaragua.
  Fact 0.78: Tecpan Guatemala (Chimaltenango, Guatemala) was founded in 1524.
  Fact 0.79: Tecpan Guatemala was the first capital of Guatemala.
  Fact 0.80: Tecpan Guatemala is the oldest continuously inhabited European established settlement in Guatemala.
  Fact 0.81: Trujillo (Colon, Honduras) was founded in 1524.
  Fact 0.82: Trujillo is the oldest continuously inhabited European established settlement in Honduras.
  Fact 0.83: Santa Marta (Magdalena, Colombia) was founded in 1525.
  Fact 0.84: Santa Marta is the oldest continuously inhabited European established settlement in Colombia.
  Fact 0.85: Acapulco (Guerrero, Mexico) was founded in 1525.
  Fact 0.86: San Salvador (San Salvador, El Salvador) was founded in 1525.
  Fact 0.87: San Salvador was rebuilt and changed locations twice afterwards, in 1528 and 1545.
  Fact 0.88: San Salvador is the oldest continuously inhabited European established settlement in El Salvador.
  Fact 0.89: Acambaro (Guanajuato, Mexico) was founded in 1526.
  Fact 0.90: Chimaltenango (Chimaltenango, Guatemala) was founded in 1526.
  Fact 0.91: San Cristobal de las Casas (Chiapas, Mexico) was founded in 1528.
  Fact 0.92: San Cristobal de las Casas was originally called Villa Real de Chiapa.
  Fact 0.93: Camaguey (Camaguey Province, Cuba) was founded in 1528.
  Fact 0.94: Camaguey was founded in 1515 on the northern coast as Puerto del Principe.
  Fact 0.95: Camaguey was moved in 1528 to its present location.
  Fact 0.96: Maracaibo (Zulia, Venezuela) was founded in 1529.
  Fact 0.97: San Blas (Nayarit, Mexico) was founded in 1530 by Nuño Beltrán.
  Fact 0.98: Puebla (Puebla, Mexico) was founded in 1531 by Fr. Toribio de Benavente (Motolinia) on 16 April.
  Fact 0.99: Oaxaca (Oaxaca, Mexico) was founded in 1532.
  Fact 0.100: Piura (Piura, Peru) was founded in 1532.
  Fact 0.101: Sao Vicente (Sao Paulo, Brazil) was founded in 1532.
  Fact 0.102: Sao Vicente is the oldest continuously inhabited European established settlement in Brazil.
  Fact 0.103: Sao Vicente is the oldest Portuguese established settlement in the New World.
  Fact 0.104: Tepic (Nayarit, Mexico) was founded in 1532 as capital of Nueva Galicia.
  Fact 0.105: Cananeia (Sao Paulo, Brazil) was founded in 1533.
  Fact 0.106: Cananeia is the second oldest city in Brazil.
  Fact 0.107: Cartagena de Indias (Bolivar, Colombia) was founded in 1533.
  Fact 0.108: Otavalo (Imbabura, Ecuador) was founded in 1534.
  Fact 0.109: Quito (Pichincha, Ecuador) was founded in 1534.
  Fact 0.110: Quito is the oldest continuously inhabited European established settlement in Ecuador.
  Fact 0.111: Spanish Town (Saint Catherine, Jamaica) was founded in 1534 by the Spanish as Santiago de la Vega or St. Jago de la Vega.
  Fact 0.112: Spanish Town is the oldest continuously inhabited European established settlement in Jamaica.
  Fact 0.113: Trujillo (La Libertad, Peru) was founded in 1534.
  Fact 0.114: Trujillo is the oldest continuously inhabited European established settlement in Peru.
  Fact 0.115: Lima (Lima, Peru) was founded in 1535.
  Fact 0.116: Igarassu (Pernambuco, Brazil) was the site of the first European settlement in Brazil, the feitoria of Igarassu, in 1516.
  Fact 0.117: Igarassu was founded in 1535.
  Fact 0.118: Olinda (Pernambuco, Brazil) was founded in 1535.
  Fact 0.119: Olinda is one of the best-preserved colonial cities in Brazil.
  Fact 0.120: Vila Velha (Espirito Santo, Brazil) was founded in 1535.
  Fact 0.121: Paria (Oruro, Bolivia) was founded in 1535.
  Fact 0.122: Paria was the first Spanish settlement in Bolivia.
  Fact 0.123: Santiago de Cali (Valle del Cauca, Colombia) was founded in 1536.
  Fact 0.124: Popayan (Cauca, Colombia) was founded in 1536.
  Fact 0.125: San Pedro Sula (Cortes, Honduras) was founded in 1536.
  Fact 0.126: Tupiza (Potosi, Bolivia) was founded in 1536.
  Fact 0.127: Asuncion (Paraguay) was founded in 1537.
  Fact 0.128: Asuncion is the oldest continuously inhabited European established settlement in Paraguay.
  Fact 0.129: Recife (Pernambuco, Brazil) was founded in 1537.
  Fact 0.130: Recife is the capital of the state of Pernambuco.
  Fact 0.131: Recife is the oldest city among Brazil's current state capitals.
  Fact 0.132: Bogota (Cundinamarca, Colombia) was founded in 1538.
  Fact 0.133: Guayaquil (Guayas, Ecuador) was founded in 1538.
  Fact 0.134: Iguape (Sao Paulo, Brazil) was founded in 1538.
  Fact 0.135: Sucre (Chuquisaca, Bolivia) was founded in 1538.
  Fact 0.136: Sucre was originally called Ciudad de la Plata de la Nueva Toledo.
  Fact 0.137: Sucre is the oldest continuously inhabited European established settlement in Bolivia.
  Fact 0.138: Tunja (Boyaca, Colombia) was founded in 1539.
  Fact 0.139: Velez (Santander, Colombia) was founded in 1539.
  Fact 0.140: Pasto (Narino, Colombia) was founded in 1539.
  Fact 0.141: Arequipa (Arequipa, Peru) was founded in 1540.
  Fact 0.142: Ayacucho (Ayacucho, Peru) was founded in 1540.
  Fact 0.143: Campeche (Campeche, Mexico) was founded in 1540.
  Fact 0.144: Yuriria (Guanajuato, Mexico) was founded in 1540.
  Fact 0.145: Mompox (Bolivar, Colombia) was founded in 1540.
  Fact 0.146: Supia (Caldas, Colombia) was founded in 1540.
  Fact 0.147: Campeche was founded on the site of the Mayan village Kimpech.
  Fact 0.148: Huanuco (Huanuco, Peru) was founded in 1541.
  Fact 0.149: Moquegua (Moquegua, Peru) was founded in 1541.
  Fact 0.150: Santiago (Santiago, Chile) was founded in 1541.
  Fact 0.151: Santiago is the oldest continuously inhabited European established settlement in Chile.
  Fact 0.152: Santafe de Antioquia (Antioquia, Colombia) was founded in 1541.
  Fact 0.153: Guadalajara (Jalisco, Mexico) was founded in 1542.
  Fact 0.154: Genaro Codina (Zacatecas, Mexico) was founded in 1542.
  Fact 0.155: San Miguel de Allende (Guanajuato, Mexico) was founded in 1542.
  Fact 0.156: Merida (Yucatan, Mexico) was founded in 1542.
  Fact 0.157: Merida was built on the site of the Mayan city T'ho.
  Fact 0.158: San Luis de la Paz (Guanajuato, Mexico) was founded in 1542.
  Fact 0.159: Antigua Guatemala (Sacatepequez, Guatemala) was founded in 1543.
  Fact 0.160: Coban (Alta Verapaz, Guatemala) was founded in 1543.
  Fact 0.161: Santa Cruz Verapaz (Alta Verapaz, Guatemala) was founded in 1543.
  Fact 0.162: Potosi (Potosi, Bolivia) was founded in 1545.
  Fact 0.163: San Juan de los Remedios (Villa Clara Province, Cuba) was founded in 1545.
  Fact 0.164: San Juan de los Remedios has the most uncertain foundation date in Cuba.
  Fact 0.165: The date 1545 for San Juan de los Remedios was accepted because it is when the village erected its town hall.
  Fact 0.166: The origin of San Juan de los Remedios has been documented back to 1513.
  Fact 0.167: Riohacha (La Guajira, Colombia) was founded in 1545.
  Fact 0.168: Santos (Sao Paulo, Brazil) was founded in 1546.
  Fact 0.169: Neyba (Bahoruco, Dominican Republic) was founded in 1546.
  Fact 0.170: Tepezala (Aguascalientes, Mexico) was founded in 1546.
  Fact 0.171: Asientos (Aguascalientes, Mexico) was founded in 1548.
  Fact 0.172: La Paz (La Paz, Bolivia) was founded in 1548.
  Fact 0.173: Panuco (Zacatecas, Mexico) was founded in 1548.
  Fact 0.174: Zacatecas (Zacatecas, Mexico) was founded in 1548.
  Fact 0.175: Salvador (Bahia, Brazil) was founded in 1549.
  Fact 0.176: Concepcion (Concepcion, Chile) was founded in 1550.
  Fact 0.177: Valledupar (Cesar, Colombia) was founded in 1550.
  Fact 0.178: Ibague (Tolima, Colombia) was founded in 1550.
  Fact 0.179: Vitoria (Espirito Santo, Brazil) was founded in 1551.
  Fact 0.180: Mocoa (Putumayo) was founded in 1551.
  Fact 0.181: Sonsonate (Sonsonate, El Salvador) was founded in 1552.
  Fact 0.182: Santiago del Estero (Santiago del Estero, Argentina) was founded in 1553.
  Fact 0.183: Santiago del Estero is the oldest continuously inhabited European established settlement in Argentina.
  Fact 0.184: Sao Bernardo do Campo (Sao Paulo, Brazil) was founded in 1553.
  Fact 0.185: Embu (Sao Paulo, Brazil) was founded in 1554.
  Fact 0.186: Sao Paulo (Sao Paulo, Brazil) was founded in 1554.
  Fact 0.187: Sain Alto (Zacatecas, Mexico) was founded in 1555.
  Fact 0.188: Chalchihuites (Zacatecas, Mexico) was founded in 1556.
  Fact 0.189: Cuenca (Azuay, Ecuador) was founded in 1557 as Santa Ana de los Cuatro Rios de Cuenca.
  Fact 0.190: Santo Amaro (Bahia, Brazil) was founded in 1557.
  Fact 0.191: Merida (Merida, Venezuela) was founded in 1558.
  Fact 0.192: Londres (Catamarca, Argentina) was founded in 1558.
  Fact 0.193: Londres is the second oldest continuously inhabited European established settlement in Argentina.
  Fact 0.194: Londres was refounded and relocated several times.
  Fact 0.195: Mogi das Cruzes (Sao Paulo, Brazil) was founded in 1560.
  Fact 0.196: Ipojuca (Pernambuco, Brazil) was founded in 1560.
  Fact 0.197: Santa Cruz de la Sierra (Santa Cruz, Bolivia) was founded in 1561.
  Fact 0.198: Mendoza (Mendoza, Argentina) was founded in 1561.
  Fact 0.199: San Cristobal (Tachira, Venezuela) was founded in 1561.
  Fact 0.200: Itanhaem (Sao Paulo, Brasil) was founded in 1561.
  Fact 0.201: San Juan (San Juan, Argentina) was founded in 1562.
  Fact 0.202: Rio Grande (Zacatecas, Mexico) was founded in 1562.
  Fact 0.203: Susticacan (Zacatecas, Mexico) was founded in 1562.
  Fact 0.204: Cartago (Cartago, Costa Rica) was founded in 1563.
  Fact 0.205: Cartago is the oldest continuously inhabited European established settlement in Costa Rica.
  Fact 0.206: Sao Mateus (Espirito Santo, Brazil) was founded in 1564.
  Fact 0.207: Rio de Janeiro (Rio de Janeiro, Brazil) was founded in 1565.
  Fact 0.208: Saint Augustine (Florida, United States) was founded in 1565.
  Fact 0.209: Saint Augustine is the oldest continuously inhabited European established settlement in the continental United States.
  Fact 0.210: Saint Augustine was preceded by Puerto Rican colonies, Pensacola (Florida), which was destroyed in 1559, and Fort Caroline, which was destroyed in 1565.
  Fact 0.211: Tucuman (Tucuman, Argentina) was founded in 1565.
  Fact 0.212: Caracas (Capital District, Venezuela) was founded in 1567.
  Fact 0.213: Mazapil (Zacatecas, Mexico) was founded in 1568.
  Fact 0.214: Goiana (Pernambuco, Brazil) was founded in 1568.
  Fact 0.215: Goiana was declared a vila in 1685 and a cidade in 1840.
  Fact 0.216: Valaparaiso (Zacatecas, Mexico) was founded in 1568.
  Fact 0.217: Santa Ana (Santa Ana, El Salvador) was founded in 1569.
  Fact 0.218: Santa Ana was built over the Mayan city of Sihuatehuacan.
  Fact 0.219: Jerez (Zacatecas, Mexico) was founded in 1570.
  Fact 0.220: Sombrerete (Zacatecas, Mexico) was founded in 1570.
  Fact 0.221: Cochabamba (Cochabamba, Bolivia) was founded in 1571.
  Fact 0.222: Huancavelica (Huancavelica, Peru) was founded in 1572.
  Fact 0.223: San German (Puerto Rico, United States) was founded in 1573.
  Fact 0.224: San German is the second oldest European settlement in Puerto Rico.
  Fact 0.225: Cordoba (Cordoba, Argentina) was founded in 1573.
  Fact 0.226: Santa Fe (Santa Fe, Argentina) was founded in 1573.
  Fact 0.227: Buga (Valle del Cauca, Colombia) was founded in 1573.
  Fact 0.228: Huaraz (Ancash, Peru) was founded in 1574.
  Fact 0.229: Tarija (Tarija Department, Bolivia) was founded in 1574.
  Fact 0.230: Aguascalientes (Aguascalientes, Mexico) was founded in 1575.
  Fact 0.231: Leon (Guanajuato, Mexico) was founded in 1576.
  Fact 0.232: Juayua (Sonsonate, El Salvador) was founded in 1577.
  Fact 0.233: Fort-Liberte (Nord-Est, Haiti) was founded in 1578.
  Fact 0.234: Fort-Liberte was originally called Bayaha by the Spanish.
  Fact 0.235: Fort-Liberte is the oldest continuously inhabited European established settlement in Haiti.
  Fact 0.236: Tegucigalpa (Francisco Morazan, Honduras) was founded in 1578.
  Fact 0.237: Coamo (Puerto Rico, United States) was founded in 1579.
  Fact 0.238: Coamo is the third oldest European settlement in Puerto Rico.
  Fact 0.239: Coamo is in the same area where the Tainos had their village of Guayama.
  Fact 0.240: Coamo became officially a town in 1616.
  Fact 0.241: Coamo was given the title of villa by Spanish Royal Decree in 1778.
  Fact 0.242: Buenos Aires (Argentina) was founded in 1580.
  Fact 0.243: Santana de Parnaiba (Sao Paulo, Brasil) was founded in 1580.
  Fact 0.244: Santo Domingo (Xenacoj, Sacatepequez, Guatemala) was founded in 1580.
  Fact 0.245: Salta (Salta, Argentina) was founded in 1582.
  Fact 0.246: St John's (Newfoundland and Labrador, Canada) was founded in 1583.
  Fact 0.247: St John's is the oldest continuously inhabited British settlement in the Americas.
  Fact 0.248: Roanoke (North Carolina, United States) was founded in 1584.
  Fact 0.249: Joao Pessoa (Paraiba, Brazil) was founded in 1585.
  Fact 0.250: Concepcion del Oro (Zacatecas, Mexico) was founded in 1587.
  Fact 0.251: Corrientes (Corrientes, Argentina) was founded in 1588.
  Fact 0.252: Sao Cristovao (Sergipe, Brazil) was founded in 1590.
  Fact 0.253: Sao Cristovao was the first capital of Sergipe.
  Fact 0.254: Sao Cristovao was formerly known as Sergipe d'El Rei.
  Fact 0.255: Jimenez del Teul (Zacatecas, Mexico) was founded in 1591.
  Fact 0.256: Juan Aldama (Zacatecas, Mexico) was founded in 1591.
  Fact 0.257: Paudalho (Pernambuco, Brazil) was founded in 1591.
  Fact 0.258: La Rioja (La Rioja, Argentina) was founded in 1591.
  Fact 0.259: San Salvador de Jujuy (Jujuy, Argentina) was founded in 1593.
  Fact 0.260: Pinos (Zacatecas, Mexico) was founded in 1594.
  Fact 0.261: San Luis (San Luis, Argentina) was founded in 1594.
  Fact 0.262: Monterrey (Nuevo Leon, Mexico) was founded in 1596.
  Fact 0.263: Villahermosa (Tabasco, Mexico) was founded in 1596.
  Fact 0.264: Villahermosa was originally called Villa Felipe II.
  Fact 0.265: Portobelo (Colon, Panama) was founded in 1597.
  Fact 0.266: Parras (Coahuila, Mexico) was founded in 1598.
  Fact 0.267: Natal (Rio Grande do Norte, Brazil) was founded in 1599.
  Fact 0.268: Tadoussac (Quebec, Canada) was founded in 1599.
  Fact 0.269: Tadoussac is the oldest continuously inhabited French established settlement in the Americas.
  Fact 0.270: Tadoussac is the oldest European established settlement in Quebec.
  Fact 0.271: Facatativa (Cundinamarca, Colombia) was founded in 1600.
  Fact 0.272: Zipaquira (Cundinamarca, Colombia) was founded in 1600.
  Fact 0.273: David (Chiriqui Province, Panama) was founded in 1602.
  Fact 0.274: Salamanca (Guanajuato, Mexico) was founded in 1603.
  Fact 0.275: Port Royal (Nova Scotia, Canada) was founded in 1604.
  Fact 0.276: Port Royal is the oldest continuously inhabited European established settlement in Nova Scotia.
  Fact 0.277: Bayaguana (Monte Plata, Dominican Republic) was founded in 1606.
  Fact 0.278: Ibarra (Imbabura, Ecuador) was founded in 1606.
  Fact 0.279: Oruro (Oruro, Bolivia) was founded in 1606.
  Fact 0.280: Jamestown (Virginia, United States) was founded in 1607.
  Fact 0.281: Jamestown was the first permanent English established settlement in the Americas.
  Fact 0.282: Quebec City (Quebec, Canada) was founded in 1608.
  Fact 0.283: Fortaleza (Ceara, Brazil) was founded in 1609.
  Fact 0.284: Kecoughtan (Virginia, United States) was founded in 1610.
  Fact 0.285: Leon (Leon, Nicaragua) was originally founded in 1524 on the south-western shore of Lago Managua.
  Fact 0.286: Original Leon was destroyed by earthquakes.
  Fact 0.287: Leon was resettled 20 miles west in 1610 to its present-day location.
  Fact 0.288: Leon's original location is known as the ruins of Leon Viejo.
  Fact 0.289: Santa Fe (New Mexico, United States) was founded in 1610.
  Fact 0.290: Santa Fe is the oldest continuously inhabited state or territorial capital in the continental United States.
  Fact 0.291: Harbour Grace (Newfoundland and Labrador, Canada) was founded in 1610.
  Fact 0.292: Marechal Deodoro (Alagoas, Brazil) was founded in 1611.
  Fact 0.293: Sao Luis (Maranhao, Brazil) was founded in 1612.
  Fact 0.294: Sao Luis is the capital of Maranhao.
  Fact 0.295: St. George's (Bermuda) was founded in 1612.
  Fact 0.296: St. George's is the oldest continuously inhabited English established settlement in the Americas.
  Fact 0.297: Neiva (Huila, Colombia) was founded in 1612.
  Fact 0.298: Penedo (Alagoas, Brazil) was founded in 1613.
  Fact 0.299: Hopewell (Virginia, United States) was founded in 1613.
  Fact 0.300: Hopewell is the oldest continuously inhabited English settlement in North America.
  Fact 0.301: Hopewell was founded in 1613 as Bermuda City by Thomas Dale.
  Fact 0.302: Albany (New York, United States) was founded in 1614.
  Fact 0.303: Albany is the oldest US settlement north of Virginia.
  Fact 0.304: Albany is the second oldest state or territorial capital in the continental United States.
  Fact 0.305: Albany was incorporated in 1686.
  Fact 0.306: Sirinhaem (Pernambuco, Brazil) was founded in 1614.
  Fact 0.307: Sirinhaem was declared a vila in 1627 and a cidade in 1892.
  Fact 0.308: Cabo Frio (Rio de Janeiro, Brazil) was founded in 1615.
  Fact 0.309: Taos (New Mexico, United States) was founded in 1615.
  Fact 0.310: Ipiales (Narino, Colombia) was founded in 1615.
  Fact 0.311: Belem (Para, Brazil) was founded in 1616.
  Fact 0.312: Belem is the capital of Para.
  Fact 0.313: Medellin (Antioquia, Colombia) was founded in 1616 by Francisco de Herrera y Campuzano.
  Fact 0.314: Arecibo (Puerto Rico, United States) was founded in 1616.
  Fact 0.315: Jersey City (New Jersey, United States) was founded in 1617 as Pavonia, New Netherland.
  Fact 0.316: Cabo de Santo Agostinho (Pernambuco, Brazil) was founded in 1618.
  Fact 0.317: Cabo de Santo Agostinho was declared a vila in 1811 and a cidade in 1877.
  Fact 0.318: Ojocaliente (Zacatecas, Mexico) was founded in 1620.
  Fact 0.319: Plymouth (Massachusetts, United States) was founded in 1620.
  Fact 0.320: Bucaramanga (Santander, Colombia) was founded in 1622.
  Fact 0.321: Dover (New Hampshire, United States) was founded in 1623.
  Fact 0.322: Gloucester (Massachusetts, United States) was founded in 1623.
  Fact 0.323: Old Road (Saint Christopher, Saint Kitts and Nevis) was founded in 1624.
  Fact 0.324: Old Road is the oldest continuously inhabited British established settlement in the Caribbean.
  Fact 0.325: Old Road was the first successful non-Spanish established settlement in the Caribbean.
  Fact 0.326: Villa Soriano (Soriano, Uruguay) was founded in 1624.
  Fact 0.327: Villa Soriano is the oldest continuously inhabited European established settlement in Uruguay.
  Fact 0.328: New Amsterdam (New York, United States) was founded in 1625.
  Fact 0.329: New Amsterdam is now New York City.
  Fact 0.330: Quincy (Massachusetts, United States) was founded in 1625.
  Fact 0.331: Salem (Massachusetts, United States) was founded in 1626.
  Fact 0.332: Vitoria de Santo Antao (Pernambuco, Brazil) was founded in 1626.
  Fact 0.333: Vitoria de Santo Antao was declared a vila in 1811 and a cidade in 1843.
  Fact 0.334: Basseterre (Saint Christopher, Saint Kitts and Nevis) was founded in 1627.
  Fact 0.335: Basseterre is the oldest continuously inhabited French established settlement in the Caribbean.
  Fact 0.336: Bridgetown (Saint Michael, Barbados) was founded in 1628.
  Fact 0.337: Bridgetown is the second oldest continuously inhabited English established settlement in the Caribbean.
  Fact 0.338: Boston (Massachusetts, United States) was founded in 1630.
  Fact 0.339: Paramaribo (Paramaribo, Suriname) was founded in 1630.
  Fact 0.340: Paramaribo is the oldest continuously inhabited European established settlement in Suriname.
  Fact 0.341: Speightstown (Saint Peter, Barbados) was founded in 1630.
  Fact 0.342: Biddeford (Maine, United States) was founded in 1631.
  Fact 0.343: Hidalgo del Parral (Chihuahua, Mexico) was founded in 1631.
  Fact 0.344: Saint John (New Brunswick, Canada) was founded in 1631.
  Fact 0.345: Batopilas (Chihuahua, Mexico) was founded in 1632.
  Fact 0.346: Williamsburg (Virginia, United States) was founded in 1632.
  Fact 0.347: St. John's (Antigua and Barbuda) was founded in 1632.
  Fact 0.348: Green Bay (Wisconsin, United States) was founded in 1634.
  Fact 0.349: Green Bay is the oldest continuously inhabited French established settlement in the United States.
  Fact 0.350: St. Mary's City (Maryland, United States) was founded in 1634.
  Fact 0.351: Trois-Rivieres (Quebec, Canada) was founded in 1634.
  Fact 0.352: Willemstad (Curacao, Netherlands) was founded in 1634.
  Fact 0.353: Willemstad is the oldest continuously inhabited Dutch established settlement in the Caribbean.
  Fact 0.354: Concord (Massachusetts, United States) was founded in 1635.
  Fact 0.355: Old Saybrook (Connecticut, United States) was founded in 1635.
  Fact 0.356: Old Saybrook was originally a Dutch settlement called Kievits Hoek.
  Fact 0.357: Springfield (Massachusetts, United States) was founded in 1636.
  Fact 0.358: Providence (Rhode Island, United States) was founded in 1636.
  Fact 0.359: Hartford (Connecticut, United States) was founded in 1637.
  Fact 0.360: Belize City (Belize) was founded in 1638.
  Fact 0.361: Belize City is the oldest continuously inhabited European established settlement in Belize.
  Fact 0.362: Belize City was originally a Mayan city called Holzuz.
  Fact 0.363: Fort-de-France (Martinique) was founded in 1638.
  Fact 0.364: Fort-de-France was originally called Fort Saint Louis.
  Fact 0.365: New Haven (Connecticut, United States) was founded in 1638.
  Fact 0.366: Wilmington (Delaware, United States) was founded in 1638.
  Fact 0.367: Wilmington grew from Fort Christina, part of the New Sweden colony.
  Fact 0.368: Wilmington is the oldest continuously inhabited Swedish established settlement in the Americas.
  Fact 0.369: Cambridge (Massachusetts, United States) was founded in 1638.
  Fact 0.370: Rincon de Romos (Aguascalientes, Mexico) was founded in 1639.
  Fact 0.371: St. Marks (Florida, United States) was founded in 1639 or before.
  Fact 0.372: Barnstable (Massachusetts, United States) was founded in 1639.
  Fact 0.373: Barnstable is the second-oldest municipality in the United States and the only city on Cape Cod.
  Fact 0.374: Bridgeport (Connecticut, United States) was founded in 1639.
  Fact 0.375: Guilford (Connecticut, United States) was founded in 1639.
  Fact 0.376: Guilford is the location of the Henry Whitfield House.
  Fact 0.377: The Henry Whitfield House is the oldest house in Connecticut.
  Fact 0.378: The Henry Whitfield House is the oldest stone house in New England.
  Fact 0.379: Milford (Connecticut, United States) was founded in 1639.
  Fact 0.380: Newport (Rhode Island, United States) was founded in 1639.
  Fact 0.381: Tumaco (Narino, Colombia) was founded in 1640.
  Fact 0.382: Montreal (Quebec, Canada) was founded in 1642.
  Fact 0.383: Lexington (Massachusetts, United States) was founded in 1642.
  Fact 0.384: Warwick (Rhode Island, United States) was founded in 1642.
  Fact 0.385: Basse-Terre (Guadeloupe, France) was founded in 1643.
  Fact 0.386: Basse-Terre is the territorial capital.
  Fact 0.387: Dolores Hidalgo (Guanajuato, Mexico) was founded in 1643.
  Fact 0.388: Salvatierra (Guanajuato, Mexico) was founded in 1644.
  Fact 0.389: New London (Connecticut, United States) was founded in 1646.
  Fact 0.390: Alcantara (Maranhao, Brazil) was founded in 1648.
  Fact 0.391: Paranagua (Parana, Brazil) was founded in 1648.
  Fact 0.392: Annapolis (Maryland, United States) was founded in 1649.
  Fact 0.393: Castries (Castries, Saint Lucia) was founded in 1650.
  Fact 0.394: Saint George's (Saint George, Grenada) was founded in 1650.
  Fact 0.395: Norwalk (Connecticut, United States) was founded in 1651.
  Fact 0.396: Vieux Fort (Vieux Fort, Saint Lucia) was founded in 1654.
  Fact 0.397: Ciudad Juarez (Chihuahua, Mexico) was founded in 1659.
  Fact 0.398: El Paso (Texas, United States) was founded in 1659.
  Fact 0.399: Charlestown (Nevis, St Kitts & Nevis) was founded in 1660.
  Fact 0.400: Placentia (Newfoundland and Labrador, Canada) was founded in 1660.
  Fact 0.401: Placentia was the French capital until 1713.
  Fact 0.402: Placentia was originally called Plaisance.
  Fact 0.403: Rye (New York, United States) was founded in 1660.
  Fact 0.404: Cayenne (French Guiana, France) was founded in 1664.
  Fact 0.405: Cayenne is the oldest continuously inhabited European established settlement in French Guiana.
  Fact 0.406: Quibdo (Choco, Colombia) was founded in 1664.
  Fact 0.407: Port-de-Paix (Nord-Ouest, Haiti) was founded in 1665.
  Fact 0.408: Codrington (Barbuda, Antigua and Barbuda) was founded in 1666.
  Fact 0.409: Newark (New Jersey, United States) was founded in 1666.
  Fact 0.410: Paraty (Rio de Janeiro, Brazil) was founded in 1667.
  Fact 0.411: Sault Ste. Marie (Michigan, United States) was founded in 1668.
  Fact 0.412: Sault Ste. Marie is the oldest city in Michigan.
  Fact 0.413: Boqueirao (Paraiba, Brazil) was founded in 1670.
  Fact 0.414: Cap-Haitien (Nord, Haiti) was founded in 1670.
  Fact 0.415: Cap-Haitien was originally called Cap-Francais.
  Fact 0.416: Charleston (South Carolina, United States) was founded in 1670.
  Fact 0.417: Charlotte Amalie (St. Thomas, U.S. Virgin Islands, United States) was founded in 1672.
  Fact 0.418: Charlotte Amalie is the oldest permanent European settlement on Saint Thomas island.
  Fact 0.419: Charlotte Amalie is the oldest continuously inhabited Danish established settlement in the Americas.
  Fact 0.420: Kingston (Ontario, Canada) was founded in 1673.
  Fact 0.421: Kingston grew from Fort Frontenac.
  Fact 0.422: Kingston has been continuously inhabited since 1784.
  Fact 0.423: San Jose de Gracia (Aguascalientes, Mexico) was founded in 1673.
  Fact 0.424: Worcester (Massachusetts, United States) was founded in 1673.
  Fact 0.425: Cachoeira (Bahia, Brazil) was founded in 1674.
  Fact 0.426: Waterbury (Connecticut, United States) was founded in 1674.
  Fact 0.427: Crosswicks (New Jersey, United States) was founded in 1677.
  Fact 0.428: Crosswicks is a Quaker settlement.
  Fact 0.429: Crosswicks has been continuously inhabited since its founding.
  Fact 0.430: South Orange (New Jersey, United States) was founded in 1680.
  Fact 0.431: South Orange grew from Newark, later called Orange.
  Fact 0.432: Colonia del Sacramento (Colonia, Uruguay) was founded in 1680.
  Fact 0.433: Cockburn Town (Turks and Caicos Islands) was founded in 1681.
  Fact 0.434: Philadelphia (Pennsylvania, United States) was founded in 1681.
  Fact 0.435: Norfolk (Virginia, United States) was founded in 1682.
  Fact 0.436: Sao Borja (Rio Grande do Sul, Brazil) was founded in 1682.
  Fact 0.437: Dover (Delaware, United States) was founded in 1683.
  Fact 0.438: Escada (Pernambuco, Brazil) was founded in 1685.
  Fact 0.439: Trinidad (Beni Department, Bolivia) was founded in 1686.
  Fact 0.440: New Britain (Connecticut, United States) was founded in 1687.
  Fact 0.441: Sao Luiz Gonzaga (Rio Grande do Sul, Brazil) was founded in 1687.
  Fact 0.442: Sao Miguel das Missoes (Rio Grande do Sul, Brazil) was founded in 1687.
  Fact 0.443: Sao Nicolau (Rio Grande do Sul, Brazil) was founded in 1687.
  Fact 0.444: Marigot (Saint Martin) was founded in 1689.
  Fact 0.445: San Gil (Santander, Colombia) was founded in 1689.
  Fact 0.446: Port of Spain (Trinidad, Trinidad and Tobago) was founded in 1690.
  Fact 0.447: Sao Lourenco das Missoes (Rio Grande do Sul, Brazil) was founded in 1690.
  Fact 0.448: Santa Rosa de Viterbo (Boyaca, Colombia) was founded in 1690.
  Fact 0.449: Villanueva (Zacatecas, Mexico) was founded in 1692.
  Fact 0.450: Bom Jesus da Lapa (Bahia, Brazil) was founded in 1693.
  Fact 0.451: Burlington (New Jersey, United States) was founded in 1693.
  Fact 0.452: Curitiba (Parana, Brazil) was founded in 1693.
  Fact 0.453: Kingston (St Andrew, Jamaica) was founded in 1693.
  Fact 0.454: Nassau (New Providence, Bahamas) was founded in 1695.
  Fact 0.455: Saint-Marc (Artibonite, Haiti) was founded in 1695.
  Fact 0.456: Assu (Rio Grande do Norte, Brazil) was founded in 1696.
  Fact 0.457: Assu was declared a vila in 1766 and a cidade in 1845.
  Fact 0.458: Rimouski (Quebec, Canada) was founded in 1696.
  Fact 0.459: Ambato (Tungurahua, Ecuador) was founded in 1698.
  Fact 0.460: Pensacola (Florida, United States) was founded in 1698.
  Fact 0.461: Pensacola is known for being the first European settlement in North America.
  Fact 0.462: Biloxi (Mississippi, United States) was founded in 1699.
  Fact 0.463: Baton Rouge (Louisiana, United States) was founded in 1699.
  Fact 0.464: Barreiras (Bahia, Brazil) was founded in 1700.
  Fact 0.465: Detroit (Michigan, United States) was founded in 1701.
  Fact 0.466: Sao Joao del Rei (Minas Gerais, Brazil) was founded in 1701.
  Fact 0.467: Jesus Maria (Aguascalientes, Mexico) was founded in 1702.
  Fact 0.468: Mobile (Alabama, United States) was founded in 1702.
  Fact 0.469: Kaskaskia (Illinois, United States) was founded in 1703.
  Fact 0.470: Bath (North Carolina, United States) was founded in 1705.
  Fact 0.471: Albuquerque (New Mexico, United States) was founded in 1706.
  Fact 0.472: Santo Angelo (Rio Grande do Sul, Brazil) was founded in 1706.
  Fact 0.473: Chihuahua (Chihuahua, Mexico) was founded in 1709.
  Fact 0.474: Chatham (New Jersey, United States) was founded in 1710.
  Fact 0.475: Chatham was founded on land purchased in 1680.
  Fact 0.476: New Bern (North Carolina, United States) was founded in 1710.
  Fact 0.477: New Bern is the birthplace of Pepsi.
  Fact 0.478: Beaufort (South Carolina, United States) was founded in 1711.
  Fact 0.479: Ouro Preto (Minas Gerais, Brazil) was founded in 1711.
  Fact 0.480: Natchitoches (Louisiana, United States) was founded in 1714.
  Fact 0.481: Natchitoches is the oldest settlement in the Louisiana Purchase.
  Fact 0.482: Antonina (Parana, Brazil) was founded in 1714.
  Fact 0.483: Laguna (Santa Catarina, Brazil) was founded in 1714.
  Fact 0.484: Serro (Minas Gerais, Brazil) was founded in 1714.
  Fact 0.485: Kekionga (Indiana, United States) was founded in 1715 or before.
  Fact 0.486: Kekionga is the capital of the Miami tribe.
  Fact 0.487: Natchez (Mississippi, United States) was founded in 1716.
  Fact 0.488: Natchez dates to the founding of Fort Rosalie by the French.
  Fact 0.489: Brejo do Cruz (Paraiba, Brazil) was founded in 1717.
  Fact 0.490: Zacatecas (Zacatecas, Mexico) was founded in 1717.
  Fact 0.491: Westborough (Massachusetts, United States) was founded in 1717.
  Fact 0.492: Westborough was the one hundredth town in Massachusetts.
  Fact 0.493: New Orleans (Louisiana, United States) was founded in 1718.
  Fact 0.494: Rio de Contas (Bahia, Brazil) was founded in 1718.
  Fact 0.495: San Antonio (Texas, United States) was founded in 1718.
  Fact 0.496: Tiradentes (Minas Gerais, Brazil) was founded in 1718.
  Fact 0.497: Trenton (New Jersey, United States) was founded in 1719.
  Fact 0.498: Cortazar (Guanajuato, Mexico) was founded in 1721.
  Fact 0.499: Edenton (North Carolina, United States) was founded in 1722.
  Fact 0.500: Guaynabo (Puerto Rico, United States) was founded in 1723.
  Fact 0.501: Guaynabo was declared a pueblo in 1723 and a municipio in 1768.
  Fact 0.502: Brattleboro (Vermont, United States) was founded in 1724.
  Fact 0.503: Brattleboro grew out of Fort Dummer.
  Fact 0.504: Concord (New Hampshire, United States) was founded in 1725.
  Fact 0.505: Florianopolis (Santa Catarina, Brazil) was founded in 1726.
  Fact 0.506: Montevideo (Montevideo, Uruguay) was founded in 1726.
  Fact 0.507: Cuiaba (Mato Grosso, Brazil) was founded in 1727.
  Fact 0.508: Goias (Goias, Brazil) was founded in 1727.
  Fact 0.509: Pirenopolis (Goias, Brazil) was founded in 1727.
  Fact 0.510: Fredericksburg (Virginia, United States) was founded in 1728.
  Fact 0.511: Inhambupe (Bahia, Brazil) was founded in 1728.
  Fact 0.512: Nuuk (Greenland) was founded in 1728.
  Fact 0.513: Nuuk was originally a Norse colony called Godthab.
  Fact 0.514: Nuuk is the oldest continuously inhabited European established settlement in Greenland.
  Fact 0.515: Baltimore (Maryland, United States) was founded in 1729.
  Fact 0.516: Roseau (Saint George, Dominica) was founded in 1730.
  Fact 0.517: Fredericton (New Brunswick, Canada) was founded in 1732.
  Fact 0.518: Fredericton was originally called Ste. Anne's Point.
  Fact 0.519: Vincennes (Indiana, United States) was founded in 1732.
  Fact 0.520: Morretes (Parana, Brazil) was founded in 1733.
  Fact 0.521: Pau dos Ferros (Rio Grande do Norte, Brazil) was founded in 1733.
  Fact 0.522: Richmond (Virginia, United States) was founded in 1733.
  Fact 0.523: Savannah (Georgia, United States) was founded in 1733.
  Fact 0.524: Cucuta (Norte de Santander, Colombia) was founded in 1733.
  Fact 0.525: Augusta (Georgia, United States) was founded in 1736.
  Fact 0.526: Rio Grande (Rio Grande do Sul, Brazil) was founded in 1737.
  Fact 0.527: Fort Rouge (Manitoba, Canada) was founded in 1738.
  Fact 0.528: Fort Rouge is now Winnipeg, Manitoba.
  Fact 0.529: San Jose (San Jose, Costa Rica) was founded in 1738.
  Fact 0.530: Santa Rosalia de Camargo (Chihuahua, Mexico) was founded in 1740.
  Fact 0.531: Portalegre (Rio Grande do Norte, Brazil) was founded in 1740.
  Fact 0.532: San Felipe de Puerto Plata (Puerto Plata, Dominican Republic) was founded in 1740.
  Fact 0.533: Viamao (Rio Grande do Sul, Brazil) was founded in 1741.
  Fact 0.534: Canguaretama (Rio Grande do Norte, Brazil) was founded in 1743.
  Fact 0.535: Canguaretama was declared a vila in 1858 and a cidade in 1885.
  Fact 0.536: Caico (Rio Grande do Norte, Brazil) was founded in 1748.
  Fact 0.537: Caico was declared a vila in 1788 and a cidade in 1868.
  Fact 0.538: Alexandria (Virginia, United States) was founded in 1749.
  Fact 0.539: San Ignacio de Velasco (Santa Cruz Department, Bolivia) was founded in 1749.
  Fact 0.540: Port-au-Prince (Ouest, Haiti) was founded in 1749.
  Fact 0.541: Halifax (Nova Scotia, Canada) was founded in 1749.
  Fact 0.542: Windsor (Ontario, Canada) was founded in 1749.
  Fact 0.543: Windsor is the oldest continually inhabited settlement in Canada west of Montreal.
  Fact 0.544: Manchester (New Hampshire, United States) was founded in 1751.
  Fact 0.545: Manchester is a former American textile capital.
  Fact 0.546: Manchester is named after Manchester, England.
  Fact 0.547: Buique (Pernambuco, Brazil) was founded in 1752.
  Fact 0.548: Patos (Paraiba, Brazil) was founded in 1752.
  Fact 0.549: Patos was declared a vila in 1832 and a cidade in 1903.
  Fact 0.550: Tubac (Arizona, United States) was founded in 1752.
  Fact 0.551: Augusta (Maine, United States) was founded in 1754.
  Fact 0.552: Charlotte (North Carolina, United States) was founded in 1755.
  Fact 0.553: Salto (Salto, Uruguay) was founded in 1756.
  Fact 0.554: Santa Barbara de Samana (Samana, Dominican Republic) was founded in 1756.
  Fact 0.555: Yauco (Puerto Rico, United States) was founded in 1756.
  Fact 0.556: Estancia (Sergipe, Brazil) was founded in 1757.
  Fact 0.557: Pittsburgh (Pennsylvania, United States) was founded in 1758.
  Fact 0.558: Macapa (Amapa, Brazil) was founded in 1758.
  Fact 0.559: Ares (Rio Grande do Norte, Brazil) was founded in 1760.
  Fact 0.560: Sabana de la Mar (Hato Mayor, Dominican Republic) was founded in 1760.
  Fact 0.561: Sacaba (Cochabamba Department, Bolivia) was founded in 1761.
  Fact 0.562: Kingstown (St Vincent, Saint Vincent and the Grenadines) was founded in 1762.
  Fact 0.563: Parnaiba (Piaui, Brazil) was founded in 1762.
  Fact 0.564: Shepherdstown (West Virginia, United States) was founded in 1762.
  Fact 0.565: Shepherdstown was originally known as Mecklenburg.
  Fact 0.566: Allentown (Pennsylvania, United States) was founded in 1762.
  Fact 0.567: Allentown was incorporated as Northamptontown.
  Fact 0.568: Baia da Traicao (Paraiba, Brazil) was founded in 1762.
  Fact 0.569: Burlington (Vermont, United States) was founded in 1763.
  Fact 0.570: Panfilo Natera (Zacatecas, Mexico) was founded in 1763.
  Fact 0.571: Ellsworth (Maine, United States) was founded in 1763.
  Fact 0.572: Philipsburg (Sint Maarten, Netherlands) was founded in 1763.
  Fact 0.573: St. Louis (Missouri, United States) was founded in 1763.
  Fact 0.574: Charlottetown (Prince Edward Island, Canada) was founded in 1764.
  Fact 0.575: Pointe-a-Pitre (Grand Terre, Guadeloupe) was founded in 1764.
  Fact 0.576: Port Elizabeth (Grenadines, Saint Vincent and the Grenadines) was founded in 1765.
  Fact 0.577: Portsmouth (Saint John, Dominica) was founded in 1765.
  Fact 0.578: Ceara-Mirim (Rio Grande do Norte, Brazil) was founded in 1767.
  Fact 0.579: Araioses (Maranhao, Brazil) was founded in 1767.
  Fact 0.580: Elizabethton (Tennessee, United States) was founded in 1769.
  Fact 0.581: Falmouth (Trelawny, Jamaica) was founded in 1769.
  Fact 0.582: San Diego (California, United States) was founded in 1769.
  Fact 0.583: San Diego grew from the Presidio of San Diego.
  Fact 0.584: Lapa (Parana, Brazil) was founded in 1769.
  Fact 0.585: Monterey (California, United States) was founded in 1770.
  Fact 0.586: Monterey grew from the Presidio of Monterey.
  Fact 0.587: Monterey was the original capital of California.
  Fact 0.588: Araxa (Minas Gerais, Brazil) was founded in 1770.
  Fact 0.589: Calvillo (Aguascalientes, Mexico) was founded in 1771.
  Fact 0.590: Porto Alegre (Rio Grande do Sul, Brazil) was founded in 1772.
  Fact 0.591: Tucson (Arizona, United States) was founded in 1775.
  Fact 0.592: Castro (Parana, Brazil) was founded in 1774.
  Fact 0.593: Lexington (Kentucky, United States) was founded in 1775.
  Fact 0.594: Boonesborough (Kentucky, United States) was founded in 1775.
  Fact 0.595: Boonesborough grew from Fort Boonesborough, built by pioneer Daniel Boone.
  Fact 0.596: Sao Lourenco da Mata (Pernambuco, Brazil) was founded in 1775.
  Fact 0.597: Sao Lourenco da Mata had district status in 1775, village status in 1884, and municipality status in 1890.
  Fact 0.598: San Francisco (California, United States) was founded in 1776.
  Fact 0.599: Sincelejo (Sucre, Colombia) was founded in 1776.
  Fact 0.600: San Jose (California, United States) was founded in 1777.
  Fact 0.601: San Jose was originally known as El Pueblo de San Jose de Guadalupe.
  Fact 0.602: San Jose was the first town in the Spanish colony of Nueva California, which later became Alta California.
  Fact 0.603: Monteria (Cordoba, Colombia) was founded in 1777.
  Fact 0.604: Corumba (Mato Grosso do Sul, Brazil) was founded in 1778.
  Fact 0.605: Louisville (Kentucky, United States) was founded in 1778.
  Fact 0.606: Louisville grew from Fort Nelson, established by explorer George Rogers Clark.
  Fact 0.607: San Francisco de Macoris (Duarte, Dominican Republic) was founded in 1778.
  Fact 0.608: Jonesborough (Tennessee, United States) was founded in 1779.
  Fact 0.609: Jonesborough was later organized as the lost State of Franklin, with Jonesborough as capital, in 1784.
  Fact 0.610: Nashville (Tennessee, United States) was founded in 1779.
  Fact 0.611: Nashville grew from Fort Nashborough.
  Fact 0.612: Codo (Maranhao, Brazil) was founded in 1780.
  Fact 0.613: Codo was declared a vila in 1833 and a cidade in 1896.
  Fact 0.614: Las Matas de Farfan (San Juan, Dominican Republic) was founded in 1780.
  Fact 0.615: Arauca (Arauca, Colombia) was founded in 1780.
  Fact 0.616: Montpelier (Vermont, United States) was founded in 1781.
  Fact 0.617: Los Angeles (California, United States) was founded in 1781.
  Fact 0.618: Catu (Bahia, Brazil) was founded in 1782.
  Fact 0.619: Georgetown (Demerara-Mahaica, Guyana) was founded in 1782.
  Fact 0.620: Georgetown was originally called La Nouvelle Ville.
  Fact 0.621: Vitoria da Conquista (Bahia, Brazil) was founded in 1783.
  Fact 0.622: San Fernando (Trinidad, Trinidad and Tobago) was founded in 1784.
  Fact 0.623: Harrisburg (Pennsylvania, United States) was founded in 1785.
  Fact 0.624: North Battleford (Saskatchewan, Canada) was founded in 1785 as a fur trading post.
  Fact 0.625: Battleford was the capital of the North West Territories in 1877.
  Fact 0.626: North Battleford was incorporated as a city with a population of 5000 or more in 1913.
  Fact 0.627: Asheville (North Carolina, United States) was founded in 1785.
  Fact 0.628: Gustavia (Saint Barthelemy) was founded in 1785.
  Fact 0.629: Huntington (West Virginia, United States) was founded in 1785.
  Fact 0.630: Sydney (Nova Scotia, Canada) was founded in 1785.
  Fact 0.631: Columbia (South Carolina, United States) was founded in 1786.
  Fact 0.632: Escada (Pernambuco, Brazil) was declared a vila in 1854 and a cidade in 1873.
  Fact 0.633: Florissant (Missouri, United States) was founded in 1786.
  Fact 0.634: Florissant was originally known as St. Ferdinand.
  Fact 0.635: Frankfort (Kentucky, United States) was founded in 1786.
  Fact 0.636: Portland (Maine, United States) was founded in 1786.
  Fact 0.637: Boa Ventura (Paraiba, Brazil) was founded in 1787.
  Fact 0.638: Marietta (Ohio, United States) was founded in 1788.
  Fact 0.639: Cincinnati (Ohio, United States) was founded in 1788.
  Fact 0.640: Charleston (West Virginia, United States) was founded in 1788.
  Fact 0.641: Charleston grew from Fort Lee.
  Fact 0.642: Mercedes (Soriano, Uruguay) was founded in 1788.
  Fact 0.643: Buffalo (New York, United States) was founded in 1789.
  Fact 0.644: Itambe (Pernambuco, Brazil) was founded in 1789.
  Fact 0.645: Itambe was declared a vila in 1867 and a cidade in 1879.
  Fact 0.646: Hamilton (Bermuda) was founded in 1790.
  Fact 0.647: Washington (District of Columbia, United States) was founded in 1790.
  Fact 0.648: Jacksonville (Florida, United States) was founded in 1791.
  Fact 0.649: Jacksonville was known as Cowford.
  Fact 0.650: Knoxville (Tennessee, United States) was founded in 1791.
  Fact 0.651: Bangor (Maine, United States) was founded in 1791.
  Fact 0.652: Raleigh (North Carolina, United States) was founded in 1792.
  Fact 0.653: Sherbrooke (Quebec, Canada) was founded in 1793.
  Fact 0.654: Toronto (Ontario, Canada) was founded in 1793.
  Fact 0.655: Fort Wayne (Indiana, United States) was founded in 1794.
  Fact 0.656: Edmonton (Alberta, Canada) was founded in 1795.
  Fact 0.657: Edmonton grew from Fort Edmonton.
  Fact 0.658: Erie (Pennsylvania, United States) was founded in 1795.
  Fact 0.659: Erie grew from the French Fort Presque Isle.
  Fact 0.660: Maryville (Tennessee, United States) was founded in 1795.
  Fact 0.661: Maryville grew from the American Fort Craig.
  Fact 0.662: Cleveland (Ohio, United States) was founded in 1796.
  Fact 0.663: Oranjestad (Aruba, Netherlands) was founded in 1796.
  Fact 0.664: Chaguanas (Trinidad, Trinidad and Tobago) was founded in 1797.
  Fact 0.665: Franklinton (Ohio, United States) was founded in 1797.
  Fact 0.666: Franklinton was eventually absorbed by Columbus, Ohio.
  Fact 0.667: Araruama (Rio de Janeiro, Brazil) was founded in 1799.
  Fact 0.668: Hull (Quebec, Canada) was founded in 1800.
  Fact 0.669: Hull was formerly known as Wrightville.
  Fact 0.670: Sonson (Antioquia, Colombia) was founded in 1800.
  Fact 0.671: Chapadinha (Maranhao, Brazil) was founded in 1802.
  Fact 0.672: Santa Cruz de Barahona (Barahona, Dominican Republic) was founded in 1802.
  Fact 0.673: Chicago (Illinois, United States) was founded in 1803.
  Fact 0.674: Chicago grew from Fort Dearborn.
  Fact 0.675: Huntsville (Alabama, United States) was founded in 1805.
  Fact 0.676: Hot Springs (Arkansas, United States) was founded in 1807.
  Fact 0.677: Agua Preta (Pernambuco, Brazil) was founded in 1809.
  Fact 0.678: Agua Preta was declared a vila in 1846 and a cidade in 1895.
  Fact 0.679: Guarapuava (Parana, Brazil) was founded in 1810.
  Fact 0.680: Manchester (New Hampshire, United States) was founded in 1810.
  Fact 0.681: Pilao Arcado (Bahia, Brazil) was founded in 1810.
  Fact 0.682: San Bernardino (California, United States) was founded in 1810.
  Fact 0.683: Garanhuns (Pernambuco, Brazil) was founded in 1811.
  Fact 0.684: Astoria (Oregon, United States) was founded in 1811.
  Fact 0.685: Columbus (Ohio, United States) was founded in 1812.
  Fact 0.686: Pelotas (Rio Grande do Sul, Brazil) was founded in 1812.
  Fact 0.687: Pelotas was declared a vila in 1832 and a cidade in 1835.
  Fact 0.688: Capela (Sergipe, Brazil) was founded in 1813.
  Fact 0.689: Chisec (Alta Verapaz, Guatemala) was founded in 1813.
  Fact 0.690: Barranquilla (Atlantico, Colombia) was founded in 1813.
  Fact 0.691: Areia (Paraiba, Brazil) was founded in 1815.
  Fact 0.692: Hamilton (Ontario, Canada) was founded in 1815.
  Fact 0.693: Alagoinhas (Bahia, Brazil) was founded in 1816.
  Fact 0.694: Chattanooga (Tennessee, United States) was founded in 1816.
  Fact 0.695: Chattanooga was originally named Ross's Landing.
  Fact 0.696: Saginaw (Michigan, United States) was founded in 1816.
  Fact 0.697: Savaneta (Aruba, Netherlands) was founded in 1816.
  Fact 0.698: Fort Smith (Arkansas, United States) was founded in 1817.
  Fact 0.699: Itaguai (Rio de Janeiro, Brazil) was founded in 1818.
  Fact 0.700: Montgomery (Alabama, United States) was founded in 1819.
  Fact 0.701: Montgomery is near the site of the earlier French Fort Toulouse.
  Fact 0.702: Springfield (Illinois, United States) was founded in 1819.
  Fact 0.703: Managua (Managua, Nicaragua) was founded in 1819.
  Fact 0.704: Memphis (Tennessee, United States) was founded in 1819.
  Fact 0.705: Memphis is near the site of the earlier French Fort Prudhomme.
  Fact 0.706: Palmeira (Parana, Brazil) was founded in 1819.
  Fact 0.707: Tuscaloosa (Alabama, United States) was founded in 1819.
  Fact 0.708: Tuscaloosa is a former state capital.
  Fact 0.709: Tuscaloosa grew from former Native American settlements beginning in 1580.
  Fact 0.710: Cotegipe (Bahia, Brazil) was founded in 1820.
  Fact 0.711: Little Rock (Arkansas, United States) was founded in 1821.
  Fact 0.712: Indianapolis (Indiana, United States) was founded in 1821.
  Fact 0.713: Jefferson City (Missouri, United States) was founded in 1821.
  Fact 0.714: Decatur (Alabama, United States) was founded in 1821.
  Fact 0.715: Ponta Grossa (Parana, Brazil) was founded in 1822.
  Fact 0.716: Jackson (Mississippi, United States) was founded in 1822.
  Fact 0.717: Tampa (Florida, United States) was founded in 1823.
  Fact 0.718: Tampa grew from the earlier military post Fort Brooke.
  Fact 0.719: Vancouver (Washington, United States) was founded in 1824.
  Fact 0.720: Tallahassee (Florida, United States) was founded in 1824.
  Fact 0.721: Vicksburg (Mississippi, United States) was founded in 1825.
  Fact 0.722: Grand Rapids (Michigan, United States) was founded in 1825.
  Fact 0.723: Syracuse (New York, United States) was founded in 1825 as a village.
  Fact 0.724: Syracuse received a city charter in 1847.
  Fact 0.725: Ipiranga (Parana, Brazil) was founded in 1826.
  Fact 0.726: London (Ontario, Canada) was founded in 1826.
  Fact 0.727: Bytown (Ontario, Canada) was founded in 1826.
  Fact 0.728: Bytown is now Ottawa, Ontario.
  Fact 0.729: Fort Leavenworth (Kansas, United States) was founded in 1827.
  Fact 0.730: St. Andrews (Florida, United States) was founded in 1827.
  Fact 0.731: St. Andrews is now part of Panama City.
  Fact 0.732: Key West (Florida, United States) was founded in 1828.
  Fact 0.733: Greenville (South Carolina, United States) was founded in 1831.
  Fact 0.734: Piracanjuba (Goias, Brazil) was founded in 1831.
  Fact 0.735: Goianinha (Rio Grande do Norte, Brazil) was founded in 1832.
  Fact 0.736: Touros (Rio Grande do Norte, Brazil) was founded in 1832.
  Fact 0.737: Bananeiras (Paraiba, Brazil) was founded in 1833.
  Fact 0.738: Juazeiro (Bahia, Brazil) was founded in 1833.
  Fact 0.739: Kitchener (Ontario, Canada) was founded in 1833.
  Fact 0.740: Kitchener was originally called Berlin.
  Fact 0.741: Milwaukee (Wisconsin, United States) was founded in 1833.
  Fact 0.742: Nazare da Mata (Pernambuco, Brazil) was founded in 1833.
  Fact 0.743: Nazare da Mata was declared a vila in 1833 and a cidade in 1850.
  Fact 0.744: Rio Formoso (Pernambuco, Brazil) was founded in 1833.
  Fact 0.745: Rio Formoso was declared a vila in 1833 and a cidade in 1850.
  Fact 0.746: Vassouras (Rio de Janeiro, Brazil) was founded in 1833.
  Fact 0.747: Austin (Texas, United States) was founded in 1835.
  Fact 0.748: Barra do Corda (Maranhao, Brazil) was founded in 1835.
  Fact 0.749: Curupuru (Maranhao, Brazil) was founded in 1835.
  Fact 0.750: Curupuru was declared a vila in 1841 and a cidade in 1920.
  Fact 0.751: Davenport (Iowa, United States) was founded in 1836.
  Fact 0.752: Oshawa (Ontario, Canada) was founded in 1836.
  Fact 0.753: Shreveport (Louisiana, United States) was founded in 1836.
  Fact 0.754: Madison (Wisconsin, United States) was founded in 1836.
  Fact 0.755: Tulsa (Oklahoma, United States) was founded in 1836.
  Fact 0.756: The Lochapoka named their new settlement Tulasi, meaning old town.
  Fact 0.757: Gloria do Goita (Pernambuco, Brazil) was founded in 1837.
  Fact 0.758: Gloria do Goita was declared a vila in 1877 and a cidade in 1884.
  Fact 0.759: Lansing (Michigan, United States) was founded in 1837.
  Fact 0.760: Houston (Texas, United States) was founded in 1837.
  Fact 0.761: Divina Pastora (Sergipe, Brazil) was founded in 1838.
  Fact 0.762: Kansas City (Missouri, United States) was founded in 1838.
  Fact 0.763: St Paul (Minnesota, United States) was founded in 1838.
  Fact 0.764: Santa Rita (Paraiba, Brazil) was founded in 1839.
  Fact 0.765: Santa Rita was declared a vila in 1890 and a cidade in 1924.
  Fact 0.766: Summerside (Prince Edward Island, Canada) was founded in 1840.
  Fact 0.767: Villavicencio (Meta, Colombia) was founded in 1840.
  Fact 0.768: Cedar Rapids (Iowa, United States) was founded in 1841.
  Fact 0.769: Dallas (Texas, United States) was founded in 1841.
  Fact 0.770: Piata (Bahia, Brazil) was founded in 1842.
  Fact 0.771: Salem (Oregon, United States) was founded in 1842.
  Fact 0.772: Atlanta (Georgia, United States) was founded in 1843.
  Fact 0.773: Atlanta was originally known as Terminus, later renamed Marthasville.
  Fact 0.774: Atlanta acquired the name Atlanta in 1847.
  Fact 0.775: Des Moines (Iowa, United States) was founded in 1843.
  Fact 0.776: Stanley (East Falkland, Falkland Islands) was founded in 1843.
  Fact 0.777: Uruguaiana (Rio Grande do Sul, Brazil) was founded in 1843.
  Fact 0.778: Uruguaiana was declared a vila in 1846 and a cidade in 1874.
  Fact 0.779: Victoria (British Columbia, Canada) was founded in 1843.
  Fact 0.780: Lencois (Bahia, Brazil) was founded in 1844.
  Fact 0.781: Araruna (Paraiba, Brazil) was founded in 1845.
  Fact 0.782: Portland (Oregon, United States) was founded in 1845.
  Fact 0.783: Alagoa Grande (Paraiba, Brazil) was founded in 1847.
  Fact 0.784: Alagoa Grande was declared a vila in 1865 and a cidade in 1908.
  Fact 0.785: Salt Lake City (Utah, United States) was founded in 1847.
  Fact 0.786: Salt Lake City was originally known as Great Salt Lake City.
  Fact 0.787: Caruaru (Pernambuco, Brazil) was founded in 1848.
  Fact 0.788: Caruaru was the first municipality in the Agreste region of Pernambuco state.
  Fact 0.789: Rio Verde (Goias, Brazil) was founded in 1848.
  Fact 0.790: Sacramento (California, United States) was founded in 1848.
  Fact 0.791: Provo (Utah, United States) was founded in 1849.
  Fact 0.792: Manizales (Caldas, Colombia) was founded in 1849.
  Fact 0.793: San Luis (Colorado, United States) was founded in 1851.
  Fact 0.794: Seattle (Washington, United States) was founded in 1851.
  Fact 0.795: Jerico (Antioquia, Colombia) was founded in 1851.
  Fact 0.796: Mossoro (Rio Grande do Norte, Brazil) was founded in 1852.
  Fact 0.797: Sao Jose dos Pinhais (Parana, Brazil) was founded in 1852.
  Fact 0.798: Colon (Colon, Panama) was founded in 1852.
  Fact 0.799: Olympia (Washington, United States) was founded in 1853.
  Fact 0.800: Omaha (Nebraska, United States) was founded in 1854.
  Fact 0.801: Anajatuba (Maranhao, Brazil) was founded in 1854.
  Fact 0.802: Topeka (Kansas, United States) was founded in 1854.
  Fact 0.803: Aracaju (Sergipe, Brazil) was founded in 1855.
  Fact 0.804: Lincoln (Nebraska, United States) was founded in 1856.
  Fact 0.805: Lincoln was originally called Lancaster.
  Fact 0.806: O'Fallon (Missouri, United States) was founded in 1856.
  Fact 0.807: Sioux Falls (South Dakota, United States) was founded in 1856.
  Fact 0.808: Boquim (Sergipe, Brazil) was founded in 1857.
  Fact 0.809: Carson City (Nevada, United States) was founded in 1858.
  Fact 0.810: Denver (Colorado, United States) was founded in 1858.
  Fact 0.811: Yankton (South Dakota, United States) was founded in 1859.
  Fact 0.812: Chico (California, United States) was founded in 1860.
  Fact 0.813: Boise (Idaho, United States) was founded in 1863.
  Fact 0.814: Wichita (Kansas, United States) was founded in 1863.
  Fact 0.815: Casper (Wyoming, United States) was founded in 1864.
  Fact 0.816: Casper was originally called Platte Bridge Station.
  Fact 0.817: Helena (Montana, United States) was founded in 1864.
  Fact 0.818: Salinas (California, United States) was founded in 1864.
  Fact 0.819: Prince Albert (Saskatchewan, Canada) was founded in 1866.
  Fact 0.820: Prince Albert was settled by Rev. James Nisbet.
  Fact 0.821: Prince Albert was incorporated as a city with a population of 5000 or more on October 8, 1904.
  Fact 0.822: Cheyenne (Wyoming, United States) was founded in 1867.
  Fact 0.823: Gameleira (Pernambuco, Brazil) was founded in 1867.
  Fact 0.824: Minneapolis (Minnesota, United States) was founded in 1867.
  Fact 0.825: Colinas (Maranhao, Brazil) was founded in 1868.
  Fact 0.826: Colinas was declared a vila in 1870 and a cidade in 1891.
  Fact 0.827: Phoenix (Arizona, United States) was founded in 1868.
  Fact 0.828: Afogados da Ingazeira (Pernambuco, Brazil) was founded in 1870.
  Fact 0.829: Campo Largo (Parana, Brazil) was founded in 1870.
  Fact 0.830: Puerto Limon (Limon, Costa Rica) was founded in 1870.
  Fact 0.831: Aguas Belas (Pernambuco, Brazil) was founded in 1871.
  Fact 0.832: Alto Parnaiba (Maranhao, Brazil) was founded in 1871.
  Fact 0.833: Birmingham (Alabama, United States) was founded in 1871.
  Fact 0.834: Colorado Springs (Colorado, United States) was founded in 1871.
  Fact 0.835: Fargo (North Dakota, United States) was founded in 1871.
  Fact 0.836: Imbituva (Parana, Brazil) was founded in 1871.
  Fact 0.837: Sao Joao do Triunfo (Parana, Brazil) was founded in 1871.
  Fact 0.838: Anniston (Alabama, United States) was founded in 1872.
  Fact 0.839: Bismarck (North Dakota, United States) was founded in 1872.
  Fact 0.840: Tibagi (Parana, Brazil) was founded in 1872.
  Fact 0.841: Jaboatao dos Guararapes (Pernambuco, Brazil) was founded in 1873.
  Fact 0.842: Jaboatao dos Guararapes had its first settlement in 1593, village status in 1873, and city status in 1884.
  Fact 0.843: Timbauba (Pernambuco, Brazil) was founded in 1873.
  Fact 0.844: Timbauba was declared a vila in 1879 and a cidade in 1884.
  Fact 0.845: Orlando (Florida, United States) was founded in 1875.
  Fact 0.846: Calgary (Alberta, Canada) was founded in 1875.
  Fact 0.847: Calgary grew from Fort Calgary.
  Fact 0.848: Humboldt (Saskatchewan, Canada) was founded in 1875.
  Fact 0.849: Humboldt was incorporated as a city with a population of 5000 or more on November 7, 2000.
  Fact 0.850: Boa Vista (Paraiba, Brazil) was founded in 1876.
  Fact 0.851: Cajapio (Maranhao, Brazil) was founded in 1876.
  Fact 0.852: Rapid City (South Dakota, United States) was founded in 1876.
  Fact 0.853: Billings (Montana, United States) was founded in 1877.
  Fact 0.854: Jaquimeyes (Barahona, Dominican Republic) was founded in 1877.
  Fact 0.855: Coeur d'Alene (Idaho, United States) was founded in 1878.
  Fact 0.856: Colombo (Parana, Brazil) was founded in 1878.
  Fact 0.857: Vicencia (Pernambuco, Brazil) was founded in 1879.
  Fact 0.858: Vicencia was declared a vila in 1891 and a cidade in 1928.
  Fact 0.859: Pierre (South Dakota, United States) was founded in 1880.
  Fact 0.860: Juneau (Alaska, United States) was founded in 1881.
  Fact 0.861: Moose Jaw (Saskatchewan, Canada) was founded in 1881.
  Fact 0.862: Moose Jaw was incorporated as a city with a population of 5000 or more in November 1903.
  Fact 0.863: Brandon (Manitoba, Canada) was founded in 1882.
  Fact 0.864: Mao (Valverde, Dominican Republic) was founded in 1882.
  Fact 0.865: Restauracion (Dajabon, Dominican Republic) was founded in 1882.
  Fact 0.866: Yorkton (Saskatchewan, Canada) was founded in 1882.
  Fact 0.867: Yorkton was settled by the York Farmers' Colonization Company Limited on May 12, 1882.
  Fact 0.868: Yorkton was incorporated as a city with a population of 5000 or more on February 1, 1928.
  Fact 0.869: Swift Current (Saskatchewan, Canada) was founded in 1882.
  Fact 0.870: The Canadian Pacific Railway arrived at Swift Current on December 10, 1882.
  Fact 0.871: Swift Current was incorporated as a city with a population of 5000 or more on January 15, 1914.
  Fact 0.872: Regina (Saskatchewan, Canada) was founded in 1882.
  Fact 0.873: Regina was first named Pile O' Bones and then renamed Regina.
  Fact 0.874: Regina is the capital city of Saskatchewan.
  Fact 0.875: Regina was incorporated as a city with a population of 5000 or more on June 19, 1903.
  Fact 0.876: Rio Branco (Acre, Brazil) was founded in 1882.
  Fact 0.877: Saskatoon (Saskatchewan, Canada) was founded in 1883.
  Fact 0.878: Saskatoon began as the Temperance Colony founded at Nutana.
  Fact 0.879: Riversdale, Nutana, and Saskatoon were incorporated as a city with a population of 5000 or more in 1906.
  Fact 0.880: Melfort (Saskatchewan, Canada) was founded in 1884.
  Fact 0.881: Melfort's Stoney Creek settlement began in 1884.
  Fact 0.882: Melfort was incorporated as a city with a population of 5000 or more on July 1, 1907.
  Fact 0.883: Vancouver (British Columbia, Canada) was founded in 1886.
  Fact 0.884: Gulfport (Mississippi, United States) was founded in 1887.
  Fact 0.885: Virginia Beach (Virginia, United States) was founded in 1888.
  Fact 0.886: Virginia Beach grew from the Seatack community.
  Fact 0.887: Oklahoma City (Oklahoma, United States) was founded in 1889.
  Fact 0.888: Whiting (Indiana, United States) was founded in 1889.
  Fact 0.889: Ijui (Rio Grande do Sul, Brazil) was founded in 1890.
  Fact 0.890: Aliança (Pernambuco, Brazil) was founded in 1892.
  Fact 0.891: Alianca was declared a vila in 1909 and a cidade in 1928.
  Fact 0.892: Catende (Pernambuco, Brazil) was founded in 1892 as a district of Palmares.
  Fact 0.893: Catende became an autonomous municipality in 1909.
  Fact 0.894: Estevan (Saskatchewan, Canada) was founded in 1892.
  Fact 0.895: The CPR arrived in Estevan in 1893.
  Fact 0.896: Estevan was incorporated as a city with a population of 5000 or more on March 1, 1957.
  Fact 0.897: Guayaramerin (Beni, Bolivia) was founded in 1892.
  Fact 0.898: Restauracion (Dajabon, Dominican Republic) was founded in 1892.
  Fact 0.899: Vernon (British Columbia, Canada) was founded in 1892.
  Fact 0.900: Araripina (Pernambuco, Brazil) was founded in 1893 as a district of Ouricuri.
  Fact 0.901: Araripina became an autonomous municipality in 1928.
  Fact 0.902: Catuipe (Rio Grande do Sul, Brazil) was founded in 1893.
  Fact 0.903: Riberalta (Beni, Bolivia) was founded in 1894.
  Fact 0.904: Traverse City (Michigan, United States) was founded in 1895.
  Fact 0.905: Condado (Pernambuco, Brazil) was founded in 1896.
  Fact 0.906: Condado was declared a cidade in 1958.
  Fact 0.907: Inhumas (Goias, Brazil) was founded in 1896.
  Fact 0.908: Miami (Florida, United States) was founded in 1896.
  Fact 0.909: Santa Catalina la Tinta (Alta Verapaz, Guatemala) was founded in 1896.
  Fact 0.910: Jequie (Bahia, Brazil) was founded in 1897.
  Fact 0.911: Nelson (British Columbia, Canada) was founded in 1897.
  Fact 0.912: Weyburn (Saskatchewan, Canada) was founded in 1898.
  Fact 0.913: Weyburn began as the Signal Hill settlement.
  Fact 0.914: Weyburn was incorporated as a city in 1913.
  Fact 0.915: Chetumal (Quintana Roo, Mexico) was founded in 1898.
  Fact 0.916: Chetumal was originally called Payo Obispo.
  Fact 0.917: Porto Acre (Acre, Brazil) was founded in 1899.
  Fact 0.918: Afranio (Pernambuco, Brazil) was founded in 1900 as a district of Petrolina.
  Fact 0.919: Afranio became an autonomous municipality in 1963.
  Fact 0.920: Fairbanks (Alaska, United States) was founded in 1901.
  Fact 0.921: Florencia (Caqueta, Colombia) was founded in 1902.
  Fact 0.922: Lloydminster (Saskatchewan, Canada) was founded in 1903.
  Fact 0.923: Lloydminster was a Brittania colony settled by Rev. Exton Lloyd and Rev. Isaac Barr.
  Fact 0.924: Lloydminster was incorporated as a city with a population of 5000 or more in 1958.
  Fact 0.925: Cruzeiro do Sul (Acre, Brazil) was founded in 1904.
  Fact 0.926: Grytviken (South Georgia) was founded in 1904.
  Fact 0.927: Sena Madureira (Acre, Brazil) was founded in 1904.
  Fact 0.928: Xapuri (Acre, Brazil) was founded in 1904.
  Fact 0.929: Cobija (Pando, Bolivia) was founded in 1906.
  Fact 0.930: Feijo (Acre, Brazil) was founded in 1906.
  Fact 0.931: Melville (Saskatchewan, Canada) was founded in 1907.
  Fact 0.932: Melville was incorporated as a city with a population of 5000 or more on August 1, 1960.
  Fact 0.933: Tarauaca (Acre, Brazil) was founded in 1907.
  Fact 0.934: Sao Mateus do Sul (Parana, Brazil) was founded in 1908.
  Fact 0.935: Estreito (Maranhao, Brazil) was founded in 1909.
  Fact 0.936: Brasileia (Acre, Brazil) was founded in 1910.
  Fact 0.937: Kindersley (Saskatchewan, Canada) was founded in 1910.
  Fact 0.938: Porto Walter (Acre, Brazil) was founded in 1910.
  Fact 0.939: Villazon (Potosi, Bolivia) was founded in 1910.
  Fact 0.940: Las Vegas (Nevada, United States) was founded in 1911.
  Fact 0.941: Anchorage (Alaska, United States) was founded in 1914.
  Fact 0.942: Yopal (Casanare, Colombia) was founded in 1915.
  Fact 0.943: Leticia (Amazonas, Colombia) was founded in 1928.
  Fact 0.944: Pabellon de Arteaga (Aguascalientes, Mexico) was founded in 1929.
  Fact 0.945: Puerto Carreno (Vichada, Colombia) was founded in 1930.
  Fact 0.946: Loreto (Zacatecas, Mexico) was founded in 1931.
  Fact 0.947: Yellowknife (Northwest Territories, Canada) was founded in 1935.
  Fact 0.948: Mitu (Vaupes, Colombia) was founded in 1935.
  Fact 0.949: San Jose del Guaviare (Guaviare, Colombia) was founded in 1938.
  Fact 0.950: Iqaluit (Nunavut, Canada) was founded in 1942.
  Fact 0.951: Iqaluit was originally called Frobisher Bay.
  Fact 0.952: Freeport (Grand Bahama, Bahamas) was founded in 1955.
  Fact 0.953: Ciudad del Este (Alto Parana, Paraguay) was founded in 1957.
  Fact 0.954: Ciudad del Este was originally called Puerto Flor de Lis.
  Fact 0.955: Rankin Inlet (Nunavut, Canada) was founded in 1957.
  Fact 0.956: Brasilia (Distrito Federal, Brazil) was founded in 1960.
  Fact 0.957: Brasilia was created in 1960 as the national capital.
  Fact 0.958: Belmopan (Cayo, Belize) was founded in 1970.
  Fact 0.959: Linden (Upper Demerara-Berbice, Guyana) was founded in 1970.
  Fact 0.960: Linden was formed by combining the towns of Christianburg, MacKenzie, and Wismar.
  Fact 0.961: Cancun (Quintana Roo, Mexico) was founded in 1970.
  Fact 0.962: Inirida is in Guainia, Colombia.
  Fact 0.963: Palmas (Tocantins, Brazil) was founded 1 year after the creation of the State of Tocantins.
  Fact 0.964: Palmas is the newest and smallest Brazilian capital city.

Retrieved items:
<item 1>
List of cities in the Americas by year of foundation
1664 Quibdó Chocó Colombia 1665 Port - de-Paix Nord - Ouest Haiti 1666 Codrington Barbuda Antigua and Barbuda 1666 Newark New Jersey United States 1667 Paraty
Rio de Janeiro Brazil 1668 Sault Ste . Marie Michigan United States ( 13 ) , oldest city in Michigan . 1670 Boqueirão Paraíba Brazil 1670 Cap - Haïtien Nord
Haiti Originally called `` Cap - Français '' 1670 Charleston South Carolina United States 1672 Charlotte Amalie , St. Thomas U.S. Virgin Islands United States
Oldest permanent European settlement in Saint Thomas island . Oldest continuously inhabited Danish established settlement in the Americas . 1673 Kingston
Ontario Canada Grew from Fort Frontenac . Continuously inhabited since 1784 . 1673 San José de Gracia Aguascalientes Mexico 1673 Worcester Massachusetts United
States 1674 Cachoeira Bahia Brazil 1674 Waterbury Connecticut United States 1677 Crosswicks New Jersey United States Quaker settlement . Continuously inhabited
since founding .
</item 1>

<item 2>
List of cities in the Americas by year of foundation
1627 Basseterre Saint Christopher Saint Kitts and Nevis Oldest continuously inhabited French established settlement in the Caribbean . 1628 Bridgetown Saint
Michael Barbados Second oldest continuously inhabited English established settlement in the Caribbean . 1630 Boston Massachusetts United States 1630 Paramaribo
Paramaribo Suriname Oldest continuously inhabited European established settlement in Suriname . 1630 Speightstown Saint Peter Barbados 1631 Biddeford Maine
United States 1631 Hidalgo del Parral Chihuahua Mexico 1631 Saint John New Brunswick Canada 1632 Batopilas Chihuahua Mexico 1632 Williamsburg Virginia United
States 1632 St. John 's Antigua Antigua and Barbuda 1634 Green Bay Wisconsin United States Oldest continuously inhabited French established settlement in the
United States . 1634 St. Mary 's City Maryland United States 1634 Trois - Rivières Quebec Canada 1634 Willemstad Curaçao Netherlands Oldest continuously
inhabited Dutch established settlement in the Caribbean . 1635 Concord Massachusetts United States 1635 Old Saybrook Connecticut United States Original Dutch
settlement called Kievits Hoek .
</item 2>

<item 3>
List of cities in the Americas by year of foundation
1643 Dolores Hidalgo Guanajuato Mexico 1644 Salvatierra Guanajuato Mexico 1646 New London Connecticut United States 1648 Alcântara Maranhão Brazil 1648
Paranaguá Paraná Brazil 1649 Annapolis Maryland United States 1650 Castries Castries Saint Lucia 1650 Saint George 's Saint George Grenada 1651 Norwalk
Connecticut United States 1654 Vieux Fort Vieux Fort Saint Lucia 1659 Ciudad Juárez Chihuahua Mexico 1659 El Paso Texas United States 1660 Charlestown Nevis St
Kitts & Nevis 1660 Placentia Newfoundland and Labrador Canada French Capital until 1713 , originally called Plaisance 1660 Rye New York United States 1664
Cayenne French Guiana France Oldest continuously inhabited European established settlement in French Guiana .
</item 3>

<item 4>
List of cities in the Americas by year of foundation
1570 Jerez Zacatecas Mexico 1570 Sombrerete Zacatecas Mexico 1571 Cochabamba Cochabamba Bolivia 1572 Huancavelica Huancavelica Peru 1573 San German Puerto Rico
United States Second oldest European settlement in Puerto Rico 1573 Córdoba Córdoba Argentina 1573 Santa Fé Santa Fé Argentina 1573 Buga Valle del Cauca
Colombia 1574 Huaraz Ancash Peru 1574 Tarija Tarija Department Bolivia 1575 Aguascalientes Aguascalientes Mexico 1576 León Guanajuato Mexico 1577 Juayúa
Sonsonate El Salvador 1578 Fort - Liberté Nord - Est Haiti Originally called Bayaha by the Spanish . Oldest continuously inhabited European established
settlement in Haiti . 1578 Tegucigalpa Francisco Morazán Honduras 1579 Coamo Puerto Rico United States Third oldest European settlement in Puerto Rico . In the
same area where the Taínos had had their village of Guayama .
</item 4>

<item 5>
List of cities in the Americas by year of foundation
1564 São Mateus Espírito Santo Brazil 1565 Rio de Janeiro Rio de Janeiro Brazil 1565 Saint Augustine Florida United States Oldest continuously inhabited
European established settlement in the continental United States . Preceded by Puerto Rican colonies , Pensacola , Florida , which was destroyed in 1559 , and
Fort Caroline , destroyed in 1565 . 1565 Tucumán Tucumán Argentina 1567 Caracas Capital District Venezuela 1568 Mazapil Zacatecas Mexico 1568 Goiana Pernambuco
Brazil Declared vila in 1685 ; cidade in 1840 . 1568 Valaparaíso Zacatecas Mexico 1569 Santa Ana Santa Ana El Salvador Built over Mayan city of Sihuatehuacán .
</item 5>

<item 6>
List of cities in the Americas by year of foundation
1710 New Bern North Carolina United States Birthplace of Pepsi 1711 Beaufort South Carolina United States 1711 Ouro Preto Minas Gerais Brazil 1714 Natchitoches
Louisiana United States Oldest settlement in the Louisiana Purchase 1714 Antonina Paraná Brazil 1714 Laguna Santa Catarina Brazil 1714 Serro Minas Gerais Brazil
1715 ( prior to ) Kekionga Indiana United States Capital of the Miami tribe . 1716 Natchez Mississippi United States Dates to the founding of Fort Rosalie by
the French.
</item 6>

<item 7>
List of cities in the Americas by year of foundation
States 1693 Curitiba Paraná Brazil 1693 Kingston St Andrew Jamaica 1695 Nassau New Providence Bahamas 1695 Saint - Marc Artibonite Haiti 1696 Assú Rio Grande do
Norte Brazil Declared vila in 1766 ; cidade in 1845 . 1696 Rimouski Quebec Canada 1698 Ambato Tungurahua Ecuador 1698 Pensacola Florida United States Known for
being the first European settlement in North America . 1699 Biloxi Mississippi United States 1699 Baton Rouge Louisiana United States 1700 Barreiras Bahia
Brazil 1701 Detroit Michigan United States 1701 São João del Rei Minas Gerais Brazil 1702 Jesús María Aguascalientes Mexico 1702 Mobile Alabama United States
1703 Kaskaskia Illinois United States 1705 Bath North Carolina United States 1706 Albuquerque New Mexico United States 1706 Santo Ângelo Rio Grande do Sul
Brazil 1709 Chihuahua Chihuahua Mexico 1710 Chatham New Jersey United States On land purchased in 1680 .
</item 7>

<item 8>
List of cities in the Americas by year of foundation
1529 Maracaibo Zulia Venezuela 1530 San Blas Nayarit Mexico By Nuño Beltrán ( 5 ) 1531 Puebla Puebla Mexico by Fr . Toribio de Benavente ( `` Motolinía '' ) on
16 April ( 6 ) 1532 Oaxaca Oaxaca Mexico 1532 Piura Piura Peru 1532 São Vicente São Paulo Brazil Oldest continuously inhabited European established settlement
in Brazil , oldest Portuguese established settlement in the New World . 1532 Tepic Nayarit Mexico As capital of Nueva Galicia ( 7 ) 1533 Cananéia São Paulo
Brazil Second Oldest City in Brazil 1533 Cartagena de Indias Bolívar Colombia 1534 Otavalo Imbabura Ecuador 1534 Quito Pichincha Ecuador Oldest continuously
inhabited European established settlement in Ecuador . 1534 Spanish Town Saint Catherine Jamaica Founded by the Spanish as Santiago de la Vega or St. Jago de la
Vega . Oldest continuously inhabited European established settlement in Jamaica .
</item 8>

<item 9>
List of cities in the Americas by year of foundation
1606 Bayaguana Monte Plata Dominican Republic 1606 Ibarra Imbabura Ecuador 1606 Oruro Oruro Bolivia 1607 Jamestown Virginia United States First permanent
English established settlement in the Americas . 1608 Quebec City Quebec Canada 1609 Fortaleza Ceará Brazil 1610 Kecoughtan Virginia United States 1524 León
León Nicaragua Original location ( now known as the ruins of Leon Viejo ) on the south - western shore of Lago Managua founded in 1524 , but destroyed by
earthquakes , and resettled 20 miles west , in 1610 , to its present - day location at Leon . 1610 Santa Fe New Mexico United States Oldest continuously
inhabited state or territorial capital in the continental United States . ( 8 ) 1610 Harbour Grace Newfoundland and Labrador Canada 1611 Marechal Deodoro ,
Alagoas Alagoas Brazil 1612 São Luís Maranhão Brazil Capital of Maranhão . 1612 St. George 's Bermuda Oldest continuously inhabited English established
settlement in the Americas .
</item 9>

<item 10>
List of cities in the Americas by year of foundation
Founded in 1498 , by Bartholomew Columbus . 1502 Santa Cruz del Seibo El Seibo Dominican Republic 1504 Azua de Compostela Azua Dominican Republic 1504 Santiago
de los Caballeros Santiago Dominican Republic 1505 Cotuí Sánchez Ramírez Dominican Republic 1506 Salvaleón de Higüey La Altagracia Dominican Republic 1508 Bonao
Monseñor Nouel Dominican Republic 1508 Caparra Puerto Rico United States First city founded by Europeans , although not continuously inhabited , in the United
States . Abandoned in 1521 with the removal of the capital to San Juan . 1510 Santa Maria la Antigua del Darien Urabá Colombia First city founded by Europeans
on the continent of South America . 1510 Nombre de Dios Colon Panama Oldest continuously inhabited European established settlement in Panama . 1511 Baracoa
Guantánamo Cuba Oldest continuously inhabited European established settlement in Cuba , former capital of Cuba .
</item 10>
=== END ITEM id=024010219aac1ed9 ===
