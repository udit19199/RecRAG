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

This file has 2 items.

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
