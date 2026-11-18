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

=== ITEM id=a138-cd4a8a6ba2625ba7 ===
Gold sentences and facts:
Sentence 0: City Image Country Population Year Mexico City Mexico 8,918,653 2015 New York City United States 8,550,405 2015 Los Angeles United States 3,971,883 2015 Toronto Canada 2,826,498 2015 Chicago United States 2,720,546 2015 6 Houston United States 2,296,224 2015 7 Havana Cuba 2,117,625 2015 8 Montreal Canada 1,753,034 2015 9 Ecatepec de Morelos Mexico 1,677,678 2015 10 Philadelphia United States 1,567,442 2015 11 Phoenix United States 1,563,025 2015 12 San Antonio United States 1,469,845 2015 13 Guadalajara Mexico 1,460,148 2015 14 Puebla Mexico 1,437,939 2015 15 San Diego United States 1,394,928 2015 16 Juárez Mexico 1,382,753 2015 17 León Mexico 1,349,224 2015 18 Dallas United States 1,300,092 2015 19 Tijuana Mexico 1,298,475 2015 20 Calgary Canada 1,230,915 2015 21 Tegucigalpa Honduras 1,190,230 2015 22 Zapopan Mexico 1,179,681 2015 23 Monterrey Mexico 1,109,171 2015 24 Managua Nicaragua 1,048,134 2015 25 Nezahualcóyotl Mexico 1,039,867 2015 26 San Jose United States 1,026,908 2015 27 Santo Domingo Dominican Republic 1,007,997 2015 28 Guatemala City Guatemala 994,078 2015 29 Port - au - Prince Haiti 987,310 2015 30 Naucalpan Mexico 970,012 2015 31 Ottawa Canada 956,710 2015 32 Austin United States 931,830 2015 33 Edmonton Canada 899,447 2016 34 Mérida Mexico 892,363 2015 35 Querétaro Mexico 878,931 2015 36 Toluca Mexico 873,536 2015 37 Jacksonville United States 868,031 2015 38 Chihuahua Mexico 867,736 2015 39 San Francisco United States 864,816 2015 40 Indianapolis United States 853,173 2015 41 Columbus United States 850,106 2015 42 Fort Worth United States 833,319 2015 43 Charlotte United States 827,097 2015 44 Hermosillo Mexico 819,999 2015 45 Saltillo Mexico 791,074 2015 46 Aguascalientes Mexico 785,942 2015 47 Mississauga Canada 761,300 2015 48 San Luis Potosí Mexico 759,854 2015 49 Veracruz Mexico 752,171 2015 50 San Pedro Sula Honduras 742,118 2015 51 Santiago de los Caballeros Dominican Republic 723,162 2015 52 Culiacán Mexico 720,798 2015 53 Winnipeg Canada 718,400 2015 54 Mexicali Mexico 717,628 2015 55 Cancún Mexico 711,682 2015 56 Acapulco Mexico 710,261 2015 57 Tlalnepantla Mexico 700,734 2015 58 Seattle United States 684,451 2015 59 Guadalupe Mexico 682,880 2015 60 Denver United States 682,545 2015 61 El Paso United States 681,124 2015 62 Chimalhuacán Mexico 679,811 2015 63 Detroit United States 677,116 2015 64 Washington , D.C. United States 672,228 2015 65 Kingston Jamaica 669,627 2015 66 Boston United States 667,137 2015 67 Tlaquepaque Mexico 664,193 2015 68 Memphis United States 655,770 2015 69 Nashville United States 654,610 2015 70 Torreón Mexico 653,150 2015 71 Vancouver Canada 648,608 2015 72 Portland United States 632,309 2015 73 Reynosa Mexico 631,611 2015 74 Oklahoma City United States 631,346 2015 75 Las Vegas United States 623,747 2015 76 Baltimore United States 621,849 2015 77 Brampton Canada 620,700 2016 78 Louisville United States 615,366 2015 79 Morelia Mexico 608,190 2015 80 Milwaukee United States 600,155 2015 81 Tuxtla Gutiérrez Mexico 598,710 2015 82 Apodaca Mexico 597,207 2015 83 Durango Mexico 590,893 2015 84 Villa Nueva Guatemala 564,686 2015 85 Albuquerque United States 559,121 2015 86 Hamilton Canada 556,359 2015 87 Quebec City Canada 540,994 2015 88 Tonalá Mexico 536,111 2015 89 Tucson United States 531,641 2015 90 Cuautitlán Izcalli Mexico 531,041 2015 91 Surrey Canada 526,004 2015 92 Ciudad López Mateos Mexico 523,296 2015 93 Tultitlán Mexico 520,557 2015 94 Fresno United States 520,052 2015 95 Carrefour Haiti 501,768 2015
  Fact 0.42: Charlotte is a city in the United States, ranked 43rd, with a population of 827,097 in 2015.

Retrieved items:
<item 1>
Beijing
: Doha 2021 : Eugene World 's twenty most populous metropolitan areas 1 Tokyo - Yokohama 2 Shanghai 3 Jakarta 4 Delhi 5 Seoul - Incheon 6 Karachi 7 Guangzhou 8
Beijing 9 Shenzhen 7 Mexico City 11 São Paulo 12 Lagos 13 Mumbai 14 Cairo 15 New York 16 Osaka 17 Moscow 18 Wuhan 19 Chengdu 20 Dhaka World 's fifty most -
populous urban areas Tokyo -- Yokohama ( Keihin ) Jakarta ( Jabodetabek ) Delhi Manila ( Metro Manila ) Seoul -- Incheon ( Sudogwon ) Shanghai Karachi Beijing
New York City Guangzhou -- Foshan ( Guangfo ) São Paulo Mexico City ( Valley of Mexico ) Mumbai Osaka -- Kobe -- Kyoto ( Keihanshin ) Moscow Dhaka Greater Cairo
Los Angeles Bangkok Kolkata Greater Buenos Aires Tehran Istanbul Lagos Shenzhen Rio de Janeiro Kinshasa Tianjin Paris Lima Chengdu Greater London Nagoya (
Chūkyō ) Lahore Chennai Bangalore Chicago Bogotá Ho Chi Minh City Hyderabad Dongguan Johannesburg Wuhan Taipei - Taoyuan Hangzhou Hong Kong Chongqing Ahmedabad
Kuala Lumpur ( Klang Valley ) Quanzhou VIAF
</item 1>

<item 2>
List of North American cities by population
300 2015 48 San Luis Potosí Mexico 759,854 2015 49 Veracruz Mexico 752,171 2015 50 San Pedro Sula Honduras 742,118 2015 51 Santiago de los Caballeros Dominican
Republic 723,162 2015 52 Culiacán Mexico 720,798 2015 53 Winnipeg Canada 718,400 2015 54 Mexicali Mexico 717,628 2015 55 Cancún Mexico 711,682 2015 56 Acapulco
Mexico 710,261 2015 57 Tlalnepantla Mexico 700,734 2015 58 Seattle United States 684,451 2015 59 Guadalupe Mexico 682,880 2015 60 Denver United States 682,545
2015 61 El Paso United States 681,124 2015 62 Chimalhuacán Mexico 679,811 2015 63 Detroit United States 677,116 2015 64 Washington , D.C. United States 672,228
2015 65 Kingston Jamaica 669,
</item 2>

<item 3>
List of North American cities by population
C. United States 672,228 2015 65 Kingston Jamaica 669,627 2015 66 Boston United States 667,137 2015 67 Tlaquepaque Mexico 664,193 2015 68 Memphis United States
655,770 2015 69 Nashville United States 654,610 2015 70 Torreón Mexico 653,150 2015 71 Vancouver Canada 648,608 2015 72 Portland United States 632,309 2015 73
Reynosa Mexico 631,611 2015 74 Oklahoma City United States 631,346 2015 75 Las Vegas United States 623,747 2015 76 Baltimore United States 621,849 2015 77
Brampton Canada 620,700 2016 78 Louisville United States 615,366 2015 79 Morelia Mexico 608,190 2015 80 Milwaukee United States 600,155 2015 81 Tuxtla Gutiérrez
Mexico 598,710 2015 82 Apodaca Mexico 597,207 2015 83 Durango Mexico 590,
</item 3>

<item 4>
List of North American cities by population
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
List of North American cities by population
List of North American cities by Population - wikipedia List of North American cities by Population Jump to : navigation , search For the majority of cities in
North America , the most recent official population census results , estimates or short - term projections date to 2015 . This list compiles figures for all
North American cities with a population within city limits exceeding 500,000 that year . These figures do not reflect the population of the urban agglomeration
or metropolitan area which typically do not coincide with the administrative boundaries of the city . They refer to mid-2015 populations with the following
exceptions : Mexican cities , whose figures derive from the 2015 Intercensal Survey conducted by INEGI with a reference date of 15 March 2015 ; Calgary , whose
2015 municipal census had a reference date of April 1 ; Edmonton , which conducted municipal censuses in April 2014 and April 2016 but has not reported an
intercensal estimate of its 2015 population . The April 2016 census result is given instead .
</item 5>

<item 6>
List of North American cities by population
The April 2016 census result is given instead . Contents ( hide ) 1 List 2 See also 3 Notes 4 References List ( edit ) City Image Country Population Year Mexico
City Mexico 8,918,653 2015 New York City United States 8,550,405 2015 Los Angeles United States 3,971,883 2015 Toronto Canada 2,826,498 2015 Chicago United
States 2,720,546 2015 6 Houston United States 2,296,224 2015 7 Havana Cuba 2,117,625 2015 8 Montreal Canada 1,753,034 2015 9 Ecatepec de Morelos Mexico
1,677,678 2015 10 Philadelphia United States 1,567,442 2015 11 Phoenix United States 1,563,025 2015 12 San Antonio United States 1,469,845 2015 13 Guadalajara
Mexico 1,460,148 2015 14 Puebla Mexico 1,437,939 2015 15 San Diego United States 1,394,
</item 6>

<item 7>
List of North American cities by population
437,939 2015 15 San Diego United States 1,394,928 2015 16 Juárez Mexico 1,382,753 2015 17 León Mexico 1,349,224 2015 18 Dallas United States 1,300,092 2015 19
Tijuana Mexico 1,298,475 2015 20 Calgary Canada 1,230,915 2015 21 Tegucigalpa Honduras 1,190,230 2015 22 Zapopan Mexico 1,179,681 2015 23 Monterrey Mexico
1,109,171 2015 24 Managua Nicaragua 1,048,134 2015 25 Nezahualcóyotl Mexico 1,039,867 2015 26 San Jose United States 1,026,908 2015 27 Santo Domingo Dominican
Republic 1,007,997 2015 28 Guatemala City Guatemala 994,078 2015 29 Port - au - Prince Haiti 987,310 2015 30 Naucalpan Mexico 970,
</item 7>

<item 8>
List of North American cities by population
207 2015 83 Durango Mexico 590,893 2015 84 Villa Nueva Guatemala 564,686 2015 85 Albuquerque United States 559,121 2015 86 Hamilton Canada 556,359 2015 87
Quebec City Canada 540,994 2015 88 Tonalá Mexico 536,111 2015 89 Tucson United States 531,641 2015 90 Cuautitlán Izcalli Mexico 531,041 2015 91 Surrey Canada
526,004 2015 92 Ciudad López Mateos Mexico 523,296 2015 93 Tultitlán Mexico 520,557 2015 94 Fresno United States 520,052 2015 95 Carrefour Haiti 501,768 2015
See also ( edit ) List of cities in Canada List of cities in Mexico List of the 100 largest municipalities in Canada by population List of United States cities
by population Notes ( edit ) Jump up ^ Population estimate is that of the federal entity known since 2016 as Ciudad de México and previously as the Distrito
Federal .
</item 8>

<item 9>
List of North American cities by population
310 2015 30 Naucalpan Mexico 970,012 2015 31 Ottawa Canada 956,710 2015 32 Austin United States 931,830 2015 33 Edmonton Canada 899,447 2016 34 Mérida Mexico
892,363 2015 35 Querétaro Mexico 878,931 2015 36 Toluca Mexico 873,536 2015 37 Jacksonville United States 868,031 2015 38 Chihuahua Mexico 867,736 2015 39 San
Francisco United States 864,816 2015 40 Indianapolis United States 853,173 2015 41 Columbus United States 850,106 2015 42 Fort Worth United States 833,319 2015
43 Charlotte United States 827,097 2015 44 Hermosillo Mexico 819,999 2015 45 Saltillo Mexico 791,074 2015 46 Aguascalientes Mexico 785,942 2015 47 Mississauga
Canada 761,300 2015 48 San Luis Potosí Mexico 759,
</item 9>

<item 10>
List of North American cities by population
Retrieved 2016 - 05 - 01 . ( hide ) World 's largest cities City proper World ( Capitals ) Americas ( North Latin Central South ) Asia ( Arab world ASEAN Arabia
Far East ) Europe ( European Union Nordic ) Metropolitan area World Americas ( North South West Indies ) Europe ( European Union Nordic ) Asia Africa
Transborder Aggregate urban area † World Europe ( European Union Nordic ) Asia Urban agglomeration † North America Europe ( European Union ) Asia Historical
World Europe Related articles Global city Megacity Megalopolis Conurbation Suburbs † Aggregate urban areas are entire continuous built - up areas , grouping
together contiguous cities ( known as conurbations ) , whereas urban agglomerations are the built - up areas of individual cities . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_North_American_cities_by_population&oldid=822737139 '' Categories : Lists of cities by population Lists of
cities in the Americas North America - related lists Cities in North America Talk Contents About Wikipedia Español اردو 中文 Edit links This page was last edited
on 28 January 2018 , at 05 : 43 .
</item 10>
=== END ITEM id=a138-cd4a8a6ba2625ba7 ===

=== ITEM id=a137-6a7065055cdc88cf ===
Gold sentences and facts:
Sentence 0: The song , which has been described by members of Three Dog Night as a `` kid 's song '' and a `` silly song '' , topped the main singles charts in North America , was certified gold by the RIAA , and has since been covered by multiple artists .
  Fact 1.0: Members of Three Dog Night have described the song as a 'kid's song'.

Retrieved items:
<item 1>
Joy to the World (Three Dog Night song)
Entertainment section . August 20 , 2008 . ISSN 1053 - 2889 . Retrieved April 5 , 2011 . Jump up ^ `` Jackson v. Axton '' . Law.justia.com . Retrieved 2016 - 10
- 04 . Jump up ^ `` Forum - 1970 ( ARIA Charts : Special Occasion Charts ) '' . Australian-charts.com . Archived from the original on 2016 - 10 - 20 . Retrieved
2017 - 05 - 07 . Jump up ^ `` 100 Singles '' ( PHP ) . RPM. 15 ( 15 ) . May 29 , 1971 . Retrieved April 5 , 2011 . Jump up ^ `` Musicline.de -- Three Dog Night
Single - Chartverfolgung '' ( in German ) . Media Control Charts . PhonoNet GmbH . Jump up ^ `` Dutchcharts.nl -- Three Dog Night -- Joy To The World '' ( in
Dutch ) . Single Top 100 .
</item 1>

<item 2>
Joy to the World (Three Dog Night song)
Axton wanted to persuade his record producers to record a new melody he had written and the producers asked him to sing any words to the tune . A member of
Three Dog Night said that the original lyrics to the song were `` Jeremiah was a prophet '' but `` no one liked that '' . When Hoyt Axton performed the song to
the group , two of the three main vocalists -- Danny Hutton and Cory Wells -- rejected the song , but Chuck Negron felt that the band needed a `` silly song ''
to help bring the band back together as a working unit . Negron also felt that the song `` was n't even close to our best record , but it might have been one of
our most honest . '' The song was recorded by Three Dog Night at American Recording Company , produced by Richard Podolor , and engineered by Bill Cooper .
Unlike most Three Dog Night songs recorded at that point , instead of having just the three main vocalists singing harmony , the song was recorded with all
seven members of the band singing . Drummer Floyd Sneed sings the deep lyric `` I wan na tell you '' towards the end of the song .
</item 2>

<item 3>
Joy to the World (Three Dog Night song)
. Barrie & Jenkins . ISBN 0 - 214 - 20480 - 4 . Jump up ^ https://www.lambiek.net/artists/j/johnston_daniel.htm Jump up ^
https://www.lambiek.net/artists/j/johnston_daniel.htm Jump up ^ `` Big Lots TV Commercial , ' Joy : Recliners ' Song by Three Dog Night '' . ispot.tv .
Retrieved December 24 , 2017 .
</item 3>

<item 4>
Joy to the World (Three Dog Night song)
wikipedia.org/w/index.php?title=Joy_to_the_World_(Three_Dog_Night_song)&oldid=835489072 '' Categories : 1971 singles Billboard Hot 100 number - one singles
Three Dog Night songs Dunhill Records singles Songs written by Hoyt Axton Songs about animals Novelty songs RPM Top Singles number - one singles 1970 songs
Hidden categories : Articles needing additional references from June 2008 All articles needing additional references Articles with hAudio microformats
Singlechart usages for German Singlechart usages for Dutch100 Singlechart usages for UK Singlechart usages for Billboardhot100 Singlechart called without song
Talk Contents About Wikipedia 日本 語 Edit links This page was last edited on 9 April 2018 , at 00 : 34 . Text is available under the Creative Commons Attribution
- ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered
trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
Joy to the World (Three Dog Night song)
Genre Rock Length 3 : 50 ( album ) 3 : 17 ( single ) Label Dunhill Songwriter ( s ) Hoyt Axton Producer ( s ) Richard Podolor Three Dog Night singles chronology
`` One Man Band '' ( 1970 ) `` Joy to the World '' ( 1971 ) `` Liar '' ( 1971 ) `` One Man Band '' ( 1970 ) `` Joy to the World '' ( 1971 ) `` Liar '' ( 1971 )
`` Joy to the World '' is a song written by Hoyt Axton , and made famous by the band Three Dog Night . The song is also popularly known by its opening lyric ,
`` Jeremiah was a bullfrog '' . Three Dog Night originally released the song on their fourth studio album , Naturally in November 1970 and subsequently released
an edited version of the song as a single in February 1971 .
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
Joy to the World ( Three Dog Night song ) - wikipedia Joy to the World ( Three Dog Night song ) Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
June 2008 ) ( Learn how and when to remove this template message ) `` Joy to the World '' 1971 French release Single by Three Dog Night from the album Naturally
B - side `` I Can Hear You Calling '' Released February 1971 ( 1971 - 02 ) Format 7 '' 45 RPM Recorded 1970 at American Recording Co .
</item 7>

<item 8>
Joy to the World (Three Dog Night song)
1965 -- 1975 ( 1993 ) 20th Century Masters -- The Millennium Collection : The Best of Three Dog Night ( 1999 ) Singles `` Nobody '' `` Try a Little Tenderness
'' `` One '' `` Easy to Be Hard '' `` Eli 's Coming '' `` Celebrate '' `` Mama Told Me ( Not to Come ) '' `` Out in the Country '' `` One Man Band '' `` Joy to
the World '' `` Liar '' `` An Old Fashioned Love Song '' `` Never Been to Spain '' `` The Family of Man '' `` Black & White '' `` Pieces of April '' `` Shambala
'' `` Let Me Serenade You '' `` The Show Must Go On '' `` Sure As I 'm Sittin ' Here '' `` Play Something Sweet ( Brickyard Blues ) '' `` Til the World Ends ''
`` Everybody Is a Masterpiece '' `` It 's a Jungle Out There '' Discography Book Billboard Year - End number one singles ( 1960 -- 1979 ) 1960 : `` Theme from A
Summer Place '' -- Percy Faith 1961 : `` Tossin ' and Turnin ' '' -- Bobby Lewis 1962 : ``
</item 8>

<item 9>
Joy to the World (Three Dog Night song)
ispot.tv . Retrieved December 24 , 2017 . External links ( edit ) Lyrics of this song at MetroLyrics Three Dog Night Danny Hutton Michael Allsup Paul Kingery
Pat Bautz Eddie Reasoner David Morgan Cory Wells Chuck Negron Jimmy Greenspoon Floyd Sneed Joe Schermie Skip Konte Al Ciner Gary Moon Richard Campbell Studio
albums Three Dog Night ( 1968 ) Suitable for Framing ( 1969 ) It Ai n't Easy ( 1970 ) Naturally ( 1970 ) Harmony ( 1971 ) Seven Separate Fools ( 1972 ) Cyan (
1973 ) Hard Labor ( 1974 ) Coming Down Your Way ( 1975 ) American Pastime ( 1976 ) It 's a Jungle ( 1983 ) Live albums Captured Live at the Forum ( 1969 )
Around the World with Three Dog Night ( 1973 ) Compilations Golden Bisquits ( 1971 ) Joy to the World : Their Greatest Hits ( 1975 ) The Best of 3 Dog Night (
1982 ) Celebrate : The Three Dog Night Story , 1965 -- 1975 (
</item 9>

<item 10>
Joy to the World (Three Dog Night song)
The song , which has been described by members of Three Dog Night as a `` kid 's song '' and a `` silly song '' , topped the main singles charts in North
America , was certified gold by the RIAA , and has since been covered by multiple artists . The song is featured prominently in the film The Big Chill . It is
sung by a child character at the beginning and the Three Dog Night recording is played over the end credits . It is also played at the end of every Denver
Broncos home victory . Notable playings of this song after Broncos victories included then - Chicago Bears head coach Abe Gibron 's singing along with the song
in 1973 ; and at the end of Super Bowl XXXII , played at Qualcomm Stadium in San Diego . It was also played at the end of Super Bowl XXXIII at Pro Player ( now
Hard Rock ) Stadium in Miami Gardens , Florida and Super Bowl 50 in Santa Clara , California . Contents ( hide ) 1 Background and recording 2 Charts and awards
3 In popular culture 4 References 5 External links Background and Recording ( edit ) Some of the words are nonsensical .
</item 10>
=== END ITEM id=a137-6a7065055cdc88cf ===

=== ITEM id=a199-21872926883d4a8c ===
Gold sentences and facts:
Sentence 0: No . Name Province Birth / death Term of active service Term of service as Chief Length in days of active service Length in days as Chief Appointed by Law school Position prior to appointment Codrington Edmund Carrington England 1769 - 1849 March 1801 -- 2 April 1806 March 1801 -- 2 April 1806 North -- Counsel to the East India Company Edmund Henry Lushington England 1766 - 1839 5 September 1801 -- 1809 15 April 1807 - 1809 North University of Cambridge Barrister Alexander Johnston Scotland 1775 - 1849 15 April 1807 -- 14 February 1809 6 November 1811 - 1819 6 November 1811 - 1819 671 + ? Maitland University of Göttingen Advocate Fiscal of Ceylon William Coke England 1776 - 1818 6 March 1809 -- 1 September 1818 † 6 March 1809 -- 1811 3,466 Maitland University of Oxford Advocate Fiscal of Ceylon 5 Henry Byrne 5 November 1818 -- 1819 -- -- Brownrigg Barrister 6 Ambrose Hardinge Giffard Ireland 1771 -- 1827 8 April 1819 -- 2 March 1827 8 April 1819 -- 2 March 1827 2,885 2,885 Brownrigg Trinity College , Dublin Advocate Fiscal of Ceylon 7 Richard Ottley 1782 -- 1845 8 May 1820 -- 1833 1 November 1827 -- 1833 Barnes Chief Justice of Grenada 8 Henry Matthews England 1789 -- 1828 10 October 1827 -- 1828 -- -- Barnes King 's College , Cambridge Advocate Fiscal of Ceylon 9 Charles Marshall England 1788 - 1873 14 June 1828 -- 3 March 1836 18 February 1833 -- 3 March 1836 2,850 1,109 Barnes Jesus College , Cambridge 10 William Rough England 1772 - 1838 18 February 1833 - 1838 -- -- Trinity College , Cambridge 11 William Norris 1834 -- 1837 27 April 1836 -- 1837 Wilmot - Horton Advocate Fiscal of Ceylon 12 John Jeremie Guernsey 1795 - 1841 10 March 1836 -- 1840 -- -- Wilmot - Horton Public prosecutor of Mauritius 13 John Fredrick Stoddart Malta 1805 - 1839 2 June 1838 -- 19 August 1839 † -- 443 -- University of Glasgow 14 Anthony Oliphant Scotland 1793 -- 1859 22 October 1838 - 1854 22 October 1838 - 1854 Attorney General of the Cape Colony 15 J.G. Hildebrand 23 November 1839 -- -- -- 16 William Ogle Carr England 1802 - 1856 19 December 1939 - 1856 17 April 1854 -- 1856 Stewart - Mackenzie King 's Advocate of Ceylon 17 John Stark 10 October 1840 -- -- -- Queen 's Advocate of Ceylon 18 Christopher Temple 28 January 1854 -- -- -- Judge of the District Court of Colombo 19 Paul Ivy Sterling 20 November 1855 -- -- -- Attorney General of Hong Kong 20 William Carpenter Rowe - 1860 1857 -- 1859 -- 21 Edward Shepherd Creasy England 1812 - 1878 -- 27 March 1860 - 1875 University of Cambridge 22 Richard Morgan -- -- -- Queen 's Advocate of Ceylon 23 George Lawson -- -- -- 24 Henry Byerley Thompson - 1867 1 January 1863 -- -- -- Queen 's Advocate of Ceylon 25 Charles Henry Stewart 11 January 1867 -- -- -- 26 Richard Cayley England 26 July 1874 -- 1882 1 October 1879 -- 1882 -- Queen 's Advocate of Ceylon 27 Lovell Burchett Clarence 5 July 1879 -- -- -- 28 Harry Dias Bandaranaike 5 July 1879 -- -- -- 29 William Hackett 3 February 1877 - 1877 3 February 1877 - 1877 -- 30 John Budd Phear England 18 October 1877 -- -- -- 31 Archibald Campbell Lawrie 1837 - 1914 6 July 1892 -- 1901 -- -- Judge of the District Court of Kandy 32 Jacobus de Wet South Africa 31 May 1882 -- 29 May 1883 31 May 1882 -- 29 May 1883 363 363 33 Bruce Burnside Bahamas 21 May 1883 -- 1893 21 May 1883 -- 1893 -- Queen 's Advocate of Ceylon 34 John Winfield Bonser 13 November 1893 -- 1902 13 November 1893 -- 1902 Chief Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40 Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42 Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,651 1,651 45 James Cecil Walter Pereira 4 July 1912 -- -- -- 46 James Arthur Van Langenburg -- -- -- 47 Thomas De Sampayo 1915 -- -- -- University of Cambridge Commissioner of Assize 48 George Francis Macdaniel Ennis 5 August 1912 -- -- -- 49 Walter Shaw 10 December 1914 -- -- -- Chief Justice of British Honduras 50 Gualterus Stewart Schneider -- -- -- 51 Anton Bertram 1869 - 1937 26 July 1918 -- 1925 26 July 1918 -- 1925 University of Cambridge Attorney General of Ceylon 52 Hermann Albert Loos -- -- -- 53 William Thomas Porter 12 January 1922 -- -- -- 54 Thomas Garvin 1881 - 1940 21 June 1921 -- 1935 -- -- Manning Colombo Law College Solicitor General of Ceylon 55 John Adrian St. Valentine Jayewardene 1877 -- 1927 1923 -- -- -- Manning Judge of the District Court of Colombo 56 Llwellyn Chisholm Dalton 14 March 1925 -- -- -- 57 Lewis Matthew Maartensz 18 February 1935 -- -- -- 58 Maas Thajoon Akbar 1880 - 1944 -- -- -- University of Cambridge Solicitor General of Ceylon 59 Charles Ernest St. John Branch 1865 - 1939 3 July 1925 -- 25 May 1926 3 July 1925 -- 25 May 1926 326 326 Durham University 60 Stanley Fisher 1867 - 11 December 1926 -- 1930 11 December 1926 -- 1930 Chief Justice of Trinidad and Tobago 61 Robert William Lyall Grant 17 February 1926 -- -- -- 62 Allen Drieberg 19 September 1927 -- -- -- 63 Eugene Wilfred Jayewardene Western 1874 - 1932 May 1928 -- -- -- Fletcher Member of the Colombo Municipal Council 64 Philip James Macdonell 1873 - 1940 3 October 1930 - 1936 3 October 1930 - 1936 Chief Justice of Trinidad and Tobago 65 Lucian Macull Dominic de Silva -- -- -- 66 William Edward Barber -- -- -- 67 Francis Harold Bertram 19 February 1935 -- -- -- 68 Kenneth Poyser -- -- -- 69 Francis Soertsz - 1951 12 November 1936 -- -- -- 70 Sidney Abrahams England 1885 - 1957 3 July 1936 -- 1939 3 July 1936 -- 1939 University of Cambridge 71 Francis Moseley 2 August 1936 -- -- -- 72 Vincent Fernando 11 November 1937 -- -- -- 73 Hector Hearne 21 May 1937 -- -- -- 74 Arthur Eric Keuneman 4 March 1938 -- -- -- University of Cambridge 75 Oswald Leslie De Kretser II Western 1882 - 1959 8 April 1938 -- 1945 -- -- Caldecott University College , Colombo Commissioner of Assize 76 Arthur Wijewardena 1887 - 1964 22 August 1938 -- 1950 15 January 1949 -- 1950 University of Cambridge Solicitor General of Ceylon 77 J.H.B. Nihill 18 November 1938 -- -- -- 78 John Curtois Howard 1 December 1939 -- 1945 1 December 1939 -- 1945 Attorney General of Cyprus 79 G.H.F. Cannon -- -- -- 80 Edward Jayetileke 14 May 1942 -- 11 October 1951 1950 -- 11 October 1951 3,437 Caldecott Solicitor General of Ceylon 81 Alan Rose 11 January 1945 -- 1956 11 October 1951 -- 1956 82 A.R.H. Canakeratne 1886 - 1960 28 March 1945 -- -- -- Ceylon Law College University of London Commissioner of Assize 83 Manikku Wadumestri Hendrick de Silva -- -- -- Attorney General of Ceylon 84 Felix Reginald Dias Bandaranaike I 1861 - 1947 3 September 1946 -- -- -- University of Cambridge Judge of the District Court of Colombo 85 Chellappah Nagalingam 1893 - 1958 22 July 1947 -- 1958 -- -- Ceylon Law College Attorney General of Ceylon 86 Ralph Windham 23 July 1947 -- -- -- 87 Hema Henry Basnayake 1902 - 23 October 1947 -- 3 August 1964 1 January 1956 -- 3 August 1964 6,129 3,137 Senanayake Attorney General of Ceylon 88 Edward Frederick Noel Gratiaen 22 March 1948 -- -- -- Senanayake Attorney General of Ceylon 89 E.H.T. Gunasekera 1 March 1949 -- 25 January 1962 -- 4,713 -- Senanayake 90 M.F.S. Pulle 28 September 1949 -- -- -- Senanayake 91 V.L. St. C. Swan 1 May 1950 -- -- -- Senanayake 92 H.A. de Silva 30 November 1950 -- -- -- Senanayake 93 Kairshasp Choksy Western -- -- -- 94 Henry Weerasooriya 10 July 1953 -- 24 February 1964 -- 3,881 -- Senanayake 95 K.D. de Silva 10 July 1953 -- -- -- Senanayake 96 Miliani Sansoni 10 May 1954 -- 17 November 1966 3 August 1964 -- 17 November 1966 4,574 836 Kotelawala 97 Hugh Fernando 1910 - 1976 16 May 1955 -- 17 November 1973 20 November 1966 - 17 November 1973 6,760 2,554 Kotelawala 98 Thusew Samuel Fernando 2 May 1956 -- -- -- Bandaranaike Colombo Law College Solicitor General of Sri Lanka 99 Nadaraja Sinnetamby 3 May 1956 -- -- -- Bandaranaike 100 Lionel de Silva 5 November 1959 -- -- -- Dahanayake 101 Leonard de Silva 16 August 1966 -- -- -- Senanayake 102 Henry Thambiah Northern 1906 -- 1997 1 December 1961 -- 27 May 1968 -- 2,369 -- Bandaranaike University College , Colombo Ceylon Law College University of London Commissioner of Assize 103 Kingsley Herath 25 January 1962 -- -- -- Bandaranaike 104 Asoka Abeyesundere Southern 1906 - 1988 20 April 1962 -- 3 May 1968 -- 2,205 -- Bandaranaike Ceylon Law College 105 Ponnuduraiswamy Sriskandarajah 1905 - 1968 22 June 1962 -- 8 February 1967 -- 1,692 -- Bandaranaike Ceylon University College Ceylon Law College Commissioner of Assize 106 Gardiye Punchihewage Amaraseela Silva 22 June 1962 -- 1974 1973 -- 1974 -- Bandaranaike 107 Lionel Sirimanne -- -- -- Bandaranaike 108 Vaithalingam Manicavasagar Western 1906 - 1993 17 December 1964 -- 14 December 1967 -- 1,092 -- Bandaranaike Colombo Law College Commissioner of Assize 109 Anthony Alles Central 1911 -- 2003 1964 -- 1974 -- -- Bandaranaike University of Ceylon Ceylon Law College University of London Solicitor General of Sri Lanka 110 Veeravagu Sivasubramaniam 1908 - 1985 1966 -- 1970 -- -- Senanayake University of London Commissioner of Assize 111 George Samerawickrame 18 November 1966 -- 9 April 1981 -- 5,256 -- Senanayake 112 Victor Tennekoon Central 1914 - 8 February 1967 -- 1 July 1970 1 January 1974 - 8 September 1977 1 January 1974 - 8 September 1977 1,239 + 1,346 1,346 Senanayake University of London Ceylon Law College Judge of the Court of Appeal 113 Christopher Weeramantry Western 1926 - 1967 -- 30 April 1972 -- -- Senanayake University of Ceylon University of London Commissioner of Assize 114 Oswald Leslie De Kretser III Western 1910 - 1968 -- 21 January 1972 -- -- Senanayake University College , Colombo University of London Judge of the District Court of Panadura 115 Samarapilimudalige Wijayatilake Central 1912 - 1990 1968 -- 16 May 1975 -- -- Senanayake University College , Colombo Ceylon Law College Commissioner of Assize 116 Pandita Gunawadena 1968 -- 10 February 1971 -- -- Senanayake 117 Vincent Thamotheram 1915 - 23 March 1970 -- 1980 -- -- Senanayake Ceylon University College Ceylon Law College Commissioner of Assize 118 Hector Deheragoda 22 February 1972 -- 27 December 1976 -- 1,770 -- Bandaranaike 119 Cuda Banda Walgampaya 22 February 1972 -- 3 January 1976 -- 1,411 -- Bandaranaike 120 Jaya Pathirana - 2000 22 February 1972 -- 1978 -- -- Bandaranaike 121 Dharmadasa Wimalaratne 10 April 1972 - ? 20 October 1980 -- 15 September 1985 -- 1,791 -- Bandaranaike 122 Tellipalai Rajaratnam 1920 - 1994 10 April 1972 -- 1978 -- -- Bandaranaike Commissioner of Assize 123 Donald Sirimanne 1973 -- -- -- Gopallawa 124 Cyril Udalagama 1 January 1974 -- -- -- Gopallawa 125 T.A. de S. Wijesundera 1 January 1974 -- -- -- Gopallawa 126 Malcolm Perera 1 January 1974 -- -- -- Gopallawa 127 Izadean Mohamed Ismail 1 January 1974 -- 18 December 1981 -- 2,908 -- Gopallawa 128 Joe Weereratne 1 January 1974 -- -- -- Gopallawa 129 Arulampalam Vythialingam 1 January 1974 -- -- -- Gopallawa 130 Noel Tittawela 1 January 1974 -- -- -- Gopallawa 131 Suppiah Sharvananda Northern 1923 - 2007 1 January 1974 -- 1988 29 October 1984 -- 1988 Gopallawa Colombo Law College University of London 132 Somaratne Walpita 1 January 1974 -- -- -- Gopallawa 133 Wilmot Gunasekera 2 May 1974 -- -- -- Gopallawa 134 Barnes Ratwatte II 1918 - 2004 2 December 1974 -- ? 11 April 1981 -- 22 November 1983 -- 955 -- Gopallawa Jayewardene Ceylon Law College President of the Court of Appeal of Sri Lanka 135 Raja Wanasundera Sabaragamuwa 1924 - 2012 5 June 1975 -- ? -- -- Gopallawa University College , Colombo Ceylon Law College University of London Stanford University Solicitor General of Sri Lanka 136 Percy Colin - Thomé Southern 1922 - 12 January 1976 -- 1977 29 December 1981 -- ? -- -- Gopallawa Jayewardene University of Ceylon University of Cambridge Judge of the High Court of Colombo 137 Neville Samarakoon 1919 - 1990 -- 21 October 1984 1977 -- 21 October 1984 Jayewardene University College , Colombo Colombo Law College 138 Victor Perera 24 July 1981 -- 18 August 1983 -- 755 -- Jayewardene 139 Anton Soza Western 1919 - 2013 26 January 1982 -- 3 February 1984 -- 738 -- Jayewardene University of London Judge of the Court of Appeal 140 Parinda Ranasinghe 15 November 1982 -- 1991 1988 -- 1991 Jayewardene Colombo Law College 141 Abdul Cader 8 December 1982 -- 28 May 1985 -- 902 -- Jayewardene 142 Heevi Rodrigo 8 December 1982 -- 4 December 1984 -- 727 -- Jayewardene 143 E.A.D. Atukorale 27 November 1984 -- -- -- Jayewardene 144 Herbert Thambiah Northern 1926 -- 1992 27 November 1984 -- 1992 1991 -- 1991 Jayewardene University of Ceylon Ceylon Law College Judge of the Court of Appeal 145 Lucian Hector de Alwis 27 November 1984 -- -- -- Jayewardene 146 O.S.M. Seneviratne 10 February 1986 -- -- -- Jayewardene 147 H.A.G. de Silva 10 February 1986 -- -- -- Jayewardene Mark Fernando 1941 - 2009 4 March 1988 -- 2005 -- -- Jayewardene University of Ceylon , Peradeniya A.R.B. Amerasinghe 1937 - 2013 -- -- -- A.R.N. Fernando -- -- -- Ameer Ismail 1990 -- 1994 -- -- Premadasa University of Ceylon Ceylon Law College President of the Court of Appeal of Sri Lanka Andrew Ranjan Perera -- -- -- Pathmanathan Ramanathan 1932 - 2006 -- -- -- University of Wales President of the Court of Appeal of Sri Lanka K.M.M.B. Kulatunge -- -- -- G.P.S. de Silva -- 1991 -- 1999 Sarath N. Silva 1995 -- 16 September 1999 16 September 1999 -- 7 June 2009 3,552 Kumaratunga Sri Lanka Law College University of Brussels President of the Court of Appeal of Sri Lanka Shirani Bandaranayake North Western 1958 - 30 October 1996 -- 13 January 2013 28 January 2015 -- 29 January 2015 18 May 2011 -- 13 January 2013 28 January 2015 -- 29 January 2015 5,919 + 1 606 + 1 Kumaratunga University of Colombo University of London Lawyer C.V. Vigneswaran Western 1939 - March 2001 -- October 2004 -- -- Kumaratunga University of Ceylon Ceylon Law College Judge of the Court of Appeal Asoka de Silva 1946 - 1 August 2001 -- 17 May 2011 8 June 2009 -- 17 May 2011 3,576 708 Kumaratunga University of Peradeniya University of Colombo University of Illinois at Urbana -- Champaign President of the Court of Appeal of Sri Lanka Nihal Jayasinghe 19 February 2003 -- 29 February 2008 -- 1,836 -- Kumaratunga University of Colombo Judge of the Court of Appeal Nimal Dissanayake -- 23 April 2008 -- -- Kumaratunga Shiranee Tilakawardane 19 February 2003 - 5 April 2014 -- 1,901 -- Kumaratunga Sri Lanka Law College President of the Court of Appeal of Sri Lanka Nimal Gamini Amaratunga January 2005 -- March 2014 -- -- Kumaratunga University of Colombo International Maritime Law Institute Judge of the Court of Appeal Saleem Marsoof February 2005 -- 2014 -- -- Kumaratunga University of Colombo , University of San Diego School of Law President of the Court of Appeal of Sri Lanka Andrew Somawansa North Western 28 May 2006 -- October 2008 -- -- Rajapaksa University of Ceylon University of Nevada President of the Court of Appeal of Sri Lanka Jagath Balapatabendi Southern -- -- -- Rajapaksa Ceylon Law College Judge of the Court of Appeal Kanagasabapathy Sripavan † 1952 -- 27 March 2008 -- present 30 January 2015 -- 28 February 2017 7003368200000000000 ♠ 3,682 7003118200000000000 ♠ 1,182 Rajapaksa Sri Lanka Law College Judge of the Court of Appeal P.A. Ratnayake 2 May 2008 -- 15 February 2013 -- 7003175100000000000 ♠ 1,751 -- Rajapaksa University of Brussels Chandra Ekanayake November 2008 -- 2016 -- -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka S.I. Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 -- present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,119 -- Rajapaksa Sri Lanka Law College University of Leicester Attorney General of Sri Lanka Mohan Peiris ♠ 15 January 2013 -- 28 January 2015 15 January 2013 -- 28 January 2015 743 743 Rajapaksa Sri Lanka Law College Attorney General of Sri Lanka Rohini Marasinghe 26 April 2013 -- 2015 -- -- Rajapaksa University of Hartford Judge of the Court of Appeal Buwaneka Aluwihare † 4 December 2013 -- present -- 7003160400000000000 ♠ 1,604 -- Rajapaksa University of London Additional Solicitor General of Sri Lanka Sisira de Abrew † Western 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne † 30 January 2015 -- present -- 7003118200000000000 ♠ 1,182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present -- 7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 -- present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠ 679 -- Sirisena University of Colombo Judge of the Court of Appeal
  Fact 0.265: Samarapilimudalige Wijayatilake was from the Central province, born 1912 and died 1990.

Retrieved items:
<item 1>
List of Justices of the Supreme Court of Sri Lanka
List of justices of the Supreme Court of Sri Lanka - wikipedia List of justices of the Supreme Court of Sri Lanka Jump to : navigation , search Shirani
Bandaranayake was the 43rd Chief Justice of Sri Lanka and the first and only woman to hold the position . List of Justices of the Supreme Court of Sri Lanka In
this list , justices are placed in the order in which they took the judicial oath of office and thereby started their term of office . While many of the
justices ' positions prior to appointment are simply listed as `` lawyer '' , many had part - time positions , such as teaching , or acted as counsel to various
levels of government . Contents ( hide ) 1 Justices 2 See also 3 Notes 4 References 5 Bibliography Justices ( edit ) † Denotes current justices Began as
Provisional or Acting Puisne Justice ♠ Tenure demoted to de facto † Died in office No .
</item 1>

<item 2>
List of Justices of the Supreme Court of Sri Lanka
450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri
Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College
University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne †
30 January 2015 -- present -- 7003118200000000000 ♠ 1,
</item 2>

<item 3>
List of Justices of the Supreme Court of Sri Lanka
Jump up ^ Current to most recent page update . Bibliography ( edit ) Amerasinghe , A.R.B. ( 1986 ) . The Supreme Court of Sri Lanka : the first 185 years .
Ratmalana , Sri Lanka : Sarvodaya Book Pub . Services . ISBN 955599000X . ( hide ) Supreme Court of Sri Lanka The Court Decisions Procedure History Court
Building Current membership Chief Justice Kanagasabapathy Sripavan Puisne Justices Priyasath Dep Eva Wansundera Buwaneka Aluwihare Sisira De Abrew Priyantha
Jayawardena Upaly Abeyratne Anil Goonaratne K.T.
</item 3>

<item 4>
List of Justices of the Supreme Court of Sri Lanka
Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry
Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40
Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42
Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph
Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,
</item 4>

<item 5>
List of Justices of the Supreme Court of Sri Lanka
182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present --
7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 --
present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠
679 -- Sirisena University of Colombo Judge of the Court of Appeal See also ( edit ) List of Chief Justices of Sri Lanka List of Justices of the Supreme Court
of Sri Lanka by court composition Notes ( edit ) Jump up ^ Provisional Chief Justice Jump up ^ On leave ( 1 January - 28 April 1871 ) Jump up ^ Acting Chief
Justice ( 1 - 10 October 1879 ,
</item 5>

<item 6>
List of Justices of the Supreme Court of Sri Lanka
Canakeratne 1886 - 1960 28 March 1945 -- -- -- Ceylon Law College University of London Commissioner of Assize 83 Manikku Wadumestri Hendrick de Silva -- -- --
Attorney General of Ceylon 84 Felix Reginald Dias Bandaranaike I 1861 - 1947 3 September 1946 -- -- -- University of Cambridge Judge of the District Court of
Colombo 85 Chellappah Nagalingam 1893 - 1958 22 July 1947 -- 1958 -- -- Ceylon Law College Attorney General of Ceylon 86 Ralph Windham 23 July 1947 -- -- -- 87
Hema Henry Basnayake 1902 - 23 October 1947 -- 3 August 1964 1 January 1956 -- 3 August 1964 6,129 3,137 Senanayake Attorney General of Ceylon 88 Edward
Frederick Noel Gratiaen 22 March 1948 -- -- -- Senanayake Attorney General of Ceylon 89 E.H.T.
</item 6>

<item 7>
List of Justices of the Supreme Court of Sri Lanka
University of San Diego School of Law President of the Court of Appeal of Sri Lanka Andrew Somawansa North Western 28 May 2006 -- October 2008 -- -- Rajapaksa
University of Ceylon University of Nevada President of the Court of Appeal of Sri Lanka Jagath Balapatabendi Southern -- -- -- Rajapaksa Ceylon Law College
Judge of the Court of Appeal Kanagasabapathy Sripavan † 1952 -- 27 March 2008 -- present 30 January 2015 -- 28 February 2017 7003368200000000000 ♠ 3,682
7003118200000000000 ♠ 1,182 Rajapaksa Sri Lanka Law College Judge of the Court of Appeal P.A. Ratnayake 2 May 2008 -- 15 February 2013 -- 7003175100000000000 ♠
1,751 -- Rajapaksa University of Brussels Chandra Ekanayake November 2008 -- 2016 -- -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri
Lanka S.I.
</item 7>

<item 8>
List of Justices of the Supreme Court of Sri Lanka
Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya
Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 --
present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of
Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,
</item 8>

<item 9>
List of Justices of the Supreme Court of Sri Lanka
Kulatunge -- -- -- G.P.S. de Silva -- 1991 -- 1999 Sarath N. Silva 1995 -- 16 September 1999 16 September 1999 -- 7 June 2009 3,552 Kumaratunga Sri Lanka Law
College University of Brussels President of the Court of Appeal of Sri Lanka Shirani Bandaranayake North Western 1958 - 30 October 1996 -- 13 January 2013 28
January 2015 -- 29 January 2015 18 May 2011 -- 13 January 2013 28 January 2015 -- 29 January 2015 5,919 + 1 606 + 1 Kumaratunga University of Colombo University
of London Lawyer C.V. Vigneswaran Western 1939 - March 2001 -- October 2004 -- -- Kumaratunga University of Ceylon Ceylon Law College Judge of the Court of
Appeal Asoka de Silva 1946 - 1 August 2001 -- 17 May 2011 8 June 2009 -- 17 May 2011 3,
</item 9>

<item 10>
List of Justices of the Supreme Court of Sri Lanka
9 January - 5 June 1888 ) Jump up ^ On leave ( 25 November 1890 - February 1891 ) Jump up ^ Acting Justice ( 12 May 1903 -- 28 November 1910 ) References ( edit
) Jump up ^ Amerasinghe , A. Ranjit B ( 1986 ) , The Supreme Court of Sri Lanka : the first 185 years , Sarvodaya Book Pub . Services , ISBN 978 - 955 - 599 -
000 - 4 Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^
Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most
recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update
. Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update .
</item 10>
=== END ITEM id=a199-21872926883d4a8c ===

=== ITEM id=a184-ba4c89af2550bda0 ===
Gold sentences and facts:
Sentence 0: Country Name of animal Scientific name Pictures Ref . Algeria Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer ( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian manatee ( national aquatic animal ) Trichechus manatus Croatia Pine marten Martes martes Cuba Cuban trogon Priotelus temnurus Cyprus Cypriot mouflon Ovis orientalis Czech Republic Double - tailed lion Mythical Denmark Mute swan ( national bird ) Cygnus olor Small tortoiseshell ( national butterfly ) Aglais urticae Egypt Steppe eagle Aquila nipalensis Estonia Barn swallow ( national bird ) Hirundo rustica Eritrea Arabian camel Camelus dromedarius Ethiopia Lion Panthera leo Faroe Islands Eurasian oystercatcher ( national bird ) Haematopus ostralgus Finland Brown bear ( national animal ) Ursus arctos Whooper swan ( national bird ) Cygnus cygnus Holly blue ( national butterfly ) Celastrina argiolus European perch ( national fish ) Perca fluviatilis Seven - spot ladybird ( national insect ) Coccinella septempunctata France Gallic rooster Gallus gallus domesticus Germany Unspecified Eagle Members of Accipitridae Greece Phoenix Mythical Dolphins Delphinus Guatemala Quetzal Pharomachrus mocinno Haiti Hispaniolan Trogan Priotelus roseigaster Honduras White - tailed deer Odocoileus virginianus Hungary Turul Mythical ( most probably Falco cherrug ) Iceland Gyrfalcon Falco rusticolus India Bengal tiger ( National predator ) Panthera tigris tigris Indian peafowl ( National bird ) Pavo cristatus King cobra ( National reptile ) Ophiophagus hannah Ganges river dolphin ( National aquatic animal ) Platanista gangetica gangetica Indian elephant ( National heritage animal ) Elephas maximus indicus Indonesia Komodo dragon ( national animal ) Varanus komodoensis Garuda ( historic national animal ) Mythical Iran Persian lion Panthera leo persica Iraq Chukar partridge Alectoris chukar Goat ( KRG ) Capra aegagrus Israel Hoopoe ( national bird ) Upupa epops Jamaica Red - billed streamertail Trochilus polytmus Japan Green pheasant ( national bird ) Phasianus versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal ) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod ) Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle ( national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison
  Fact 0.74: In Macedonia, the Lion (Panthera leo) is listed as the animal in Macedonian heraldry.

Retrieved items:
<item 1>
List of national animals
bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx
leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain
as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical
United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison See also Animals portal Lists portal State
animal List of national birds Animals as heraldic charges Floral emblem National personification References Jump up ^ `` The Fennec Fox : Algeria 's National
Animal '' . algeria.com . Retrieved 2013 - 03 - 21 . Jump up ^ `` THE NATIONAL BIRD OF ANGOLA '' . BIRDS ANGOLA . Retrieved 12 April 2016 .
</item 1>

<item 2>
Graph query row:
country: China
</item 2>

<item 3>
List of national animals
Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle (
national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx
leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent
amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa
Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South
Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus
lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant
</item 3>

<item 4>
List of national animals
List of National Animals - wikipedia List of National Animals Jump to : navigation , search This article is incomplete . Please help to improve it , or discuss
the issue on the talk page . ( June 2017 ) This is a list of national animals . National animals Country Name of animal Scientific name Pictures Ref . Algeria
Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer
( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous
hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus
malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird
) Copsychus saularis Ilish ( national fish )
</item 4>

<item 5>
List of national animals
versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia
bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius
indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national
animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal
) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod )
Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus
impejanus Netherlands Lion ( heraldic symbol
</item 5>

<item 6>
List of national animals
Canadian Heritage . 2016 . Retrieved 1 December 2016 . Jump up ^ `` National Symbols of China '' . 123independenceday.com . Jump up ^ `` China National Symbols
'' . worldatlast.com. 2015 . Retrieved 2017 - 03 - 28 . Jump up ^ `` China Considers Red - crowned Crane for National Bird '' . www.china.org.cn . Jump up ^ ``
El cóndor , ave nacional - banrepcultural.org '' . Jump up ^ `` 16 de diciembre : Declaración del yigüirro como ave nacional de Costa Rica '' . Jump up ^ `` El
Venado Cola Blanca , Símbolo Nacional de la Fauna Silvestre '' . Jump up ^ `` Manatee Now Costa Rica 's New National Symbol '' . QCOSTARICA . Archived from the
original on 14 September 2014 . Jump up ^ Bousfield , Jonathan ( 2003 ) . Croatia . Rough Guides . p. 72 . ISBN 1843530848 .
</item 6>

<item 7>
List of national animals
Jump up ^ Jon Eldridge . `` Primeval Poland : where the bison roam '' . theguardian.com . Retrieved 2016 - 10 - 10 . Jump up ^ `` Poland '' . allaboutbison.com
. Retrieved 2016 - 10 - 10 . Jump up ^ Tamra Orr ( 30 June 2008 ) . Qatar . Marshall Cavendish . p. 13 . ISBN 978 - 0 - 7614 - 2566 - 3 . Retrieved 30 July 2011
. Jump up ^ `` Animalul national - Rasul - Sapte simboluri nationale ale Romaniei mai putin cunoscute '' . 9AM . Jump up ^ `` Birobidjan , Russia Beaverton 's
sister city since 1990 '' . The City of Beaverton . Retrieved 2017 - 01 - 23 . Jump up ^ `` Rwanda State Symbols , Song , Flag , and More '' . worldatlas.com .
Retrieved 2013 - 03 - 21 .
</item 7>

<item 8>
List of national animals
ISBN 978 - 0 - 313 - 34498 - 5 . Retrieved 2 January 2012 . Jump up ^ `` 12 countries that have cooler national animals than the United States '' .
globalpost.com . Retrieved 3 October 2014 . Jump up ^ `` National Animals : The Legendary , Extinct and Imaginary - 8 of 9 '' . abcnews.go.com . Retrieved 2014
- 10 - 03 . ^ Jump up to : `` Ten Unusual National Animals That Rival the Unicorn - 3 of 10 '' . Retrieved 2017 - 08 - 25 . Jump up ^ Shrestha , Nanda R. ( 2002
) . Nepal and Bangladesh : a global studies handbook . ABC - CLIO . p. 163 . ISBN 978 - 1 - 57607 - 285 - 1 . Retrieved 1 January 2012 . Jump up ^ `` Galliforms
and Pheasants '' . Archived from the original on 2007 - 01 - 12 .
</item 8>

<item 9>
List of national animals
Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New
Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus
cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra
falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National
reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy
eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao (
national animal ) Bubalus bubalis Philippine
</item 9>

<item 10>
List of national animals
Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo
Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk
Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty
animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese
dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia
Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian
manatee ( national aquatic animal ) Trichechus manatus Croatia Pine
</item 10>
=== END ITEM id=a184-ba4c89af2550bda0 ===

=== ITEM id=a153-ada74f9430a1b46b ===
Gold sentences and facts:
Sentence 0: The 1924 Winter Olympics , officially known as the I Olympic Winter Games , and known at the time as Semaine Internationale des Sports d'Hiver ( `` International Winter Sports Week '' ) , was a winter multi-sport event held in Chamonix , France , from 25 January to 5 February 1924 . Norway topped the table , collecting seventeen medals in total , including four gold , three of which were won by Thorleif Haug in the nordic combined and cross-country skiing events . Norway also achieved two podium sweeps , winning all three medals in both the 50 km cross-country skiing and the nordic combined . This remained a record at the Winter Olympics until 2014 .
  Fact 1.6: Norway collected seventeen medals in total at the 1924 Winter Olympics.

Retrieved items:
<item 1>
1924 Winter Olympics medal table
Sports Reference . Archived from the original on 25 August 2017 . Retrieved 25 August 2017 . Jump up ^ Mallon , Bill ; Heijmans , Jereon ( 2011 ) . Historical
Dictionary of the Olympic Movement . Plymouth , UK : Scarecrow Press . p. 46 . ISBN 978 - 0 - 8108 - 7522 - 7 . Archived from the original on 14 October 2017 .
Jump up ^ `` Official Report of the 1924 Olympic Games ( PDF ) ( in French ) . International Olympic Committee . 1924 . p. 662 . Archived ( PDF ) from the
original on 26 February 2016 . Jump up ^ Jorsett , Per ( 13 February 2009 ) . `` Thorleif Haug '' . Norsk biografisk leksikon ( in Norwegian ) . Oslo :
Kunnskapsforlaget . Archived from the original on 21 September 2017 . Retrieved 20 September 2017 .
</item 1>

<item 2>
1924 Winter Olympics medal table
International Olympic Committee . Archived from the original on 4 December 2008 . Retrieved 22 October 2017 . ( hide ) Olympic Games medal tables Olympic medal
All - time Olympic Games medal table Pierre de Coubertin medal Intercalated Games ( 1906 ) Summer Olympic Games 1896 1900 1904 1908 1912 1920 1924 1928 1932
1936 1948 1952 1956 1960 1964 1968 1972 1976 1980 1984 1988 1992 2000 2008 2012 2016 Winter Olympic Games 1924 1928 1932 1936 1948 1952 1956 1960 1964 1968 1972
1976 1980 1984 1988 1992 1998 2002 2006 2014 2018 Lists of Olympic medalists List of stripped Olympic medals List of Olympic medals by host nation List of ties
for medals at the Olympics Retrieved from `` https://en.wikipedia.
</item 2>

<item 3>
1924 Winter Olympics medal table
1924 Winter Olympics medal table - wikipedia 1924 Winter Olympics medal table Jump to : navigation , search Publicity poster for the event The 1924 Winter
Olympics , officially known as the I Olympic Winter Games , and known at the time as Semaine Internationale des Sports d'Hiver ( `` International Winter Sports
Week '' ) , was a winter multi-sport event held in Chamonix , France , from 25 January to 5 February 1924 . Norway topped the table , collecting seventeen
medals in total , including four gold , three of which were won by Thorleif Haug in the nordic combined and cross-country skiing events . Norway also achieved
two podium sweeps , winning all three medals in both the 50 km cross-country skiing and the nordic combined . This remained a record at the Winter Olympics
until 2014 . When it was held , the games were not formally recognised as being the Olympics , but acknowledged that they were held under the `` high patronage
of the International Olympic Committee '' . Partly due to this , varying figures are reported for the number of participants .
</item 3>

<item 4>
1924 Winter Olympics medal table
The number of silver medals is taken into consideration next and then the number of bronze medals . If nations are still tied , equal ranking is given and they
are listed alphabetically . Two bronze medals were awarded in the 500 metres speed skating event for the third place tie .
</item 4>

<item 5>
1924 Winter Olympics medal table
References ( edit ) ^ Jump up to : `` 1924 Medal Tally '' . Topend Sports . Archived from the original on 25 August 2017 . Retrieved 25 August 2017 . ^ Jump up
to : `` Chamonix 1924 '' . International Olympic Committee . Archived from the original on 1 September 2017 . Retrieved 24 August 2017 . Jump up ^ BMallon ( 18
February 2014 ) . `` Netherlands Speed Skating Medal Sweeps '' . OlympStats . Archived from the original on 25 August 2017 . Retrieved 25 August 2017 . Jump up
^ `` Bergsma breaks Olympic record to lead fourth Dutch medal sweep '' . Xinhuanet. 19 February 2014 . Archived from the original on 23 September 2015 .
Retrieved 25 August 2017 . ^ Jump up to : `` 1924 Chamonix Winter Games '' . Sports Reference . Archived from the original on 25 August 2017 .
</item 5>

<item 6>
1924 Winter Olympics medal table
Partly due to this , varying figures are reported for the number of participants . The International Olympic Committee ( IOC ) website suggests that 258
athletes from 16 nations participated in 16 events across 9 sports . In contrast , the Sports - Reference website lists 313 participants from 19 countries .
Bill Mallon , a prominent Olympic Games historian , quotes a figure of 291 competitors in his Historical Dictionary of the Olympic Movement . Meanwhile the
official report for the 1924 Summer and Winter Olympics listed 293 athletes from 17 nations . Finland placed second on the table , collecting four gold medals
amongst a total of eleven . Clas Thunberg won five of their medals , achieving a medal in each of the speed skating events he took part in ; collecting three
gold medals , one silver and one bronze . Eight of the competing nations achieved at least one gold medal , with only Belgium and the hosts , France , medalling
without winning a gold medal . One medal was reallocated in 1974 ; during the games , Thorleif Haug of Norway had been awarded the bronze medal in ski jumping ,
along with his three gold medals .
</item 6>

<item 7>
The 1924 Winter Olympics were held in Chamonix, France, from 25 January to 5 February. Norway led the medal table with 17 medals (4 gold), including three golds
for Thorleif Haug, and swept the podium in both the 50 km cross-country skiing and Nordic combined. Finland placed second with 11 medals; Clas Thunberg won five
in speed skating. Ten nations earned medals, for a total of 49. A 1974
</item 7>

<item 8>
1924 Winter Olympics medal table
Two bronze medals were awarded in the 500 metres speed skating event for the third place tie . Key * Host nation ( France ) 1924 Winter Olympics medal table
Rank NOC Gold Silver Bronze Total Norway ( NOR ) 7 6 17 Finland ( FIN ) 11 Austria ( AUT ) 0 Switzerland ( SUI ) 0 United States ( USA ) 6 Great Britain ( GBR )
7 Sweden ( SWE ) 0 8 Canada ( CAN ) 0 0 9 France ( FRA ) * 0 0 10 Belgium ( BEL ) 0 0 Total ( 10 NOCs ) 16 16 17 49 See also ( edit ) List of 1924 Winter
Olympics medal winners Notes ( edit ) Jump up ^ No country achieved more than two podium sweeps until the Netherlands in 2014 , but the record was tied prior to
that ; by Norway themselves in 1928 and 1932 , the Soviet Union in 1964 , East Germany in 1972 and 1984 , and Germany in 1998 . References ( edit ) ^ Jump up to
: `` 1924 Medal Tally '' .
</item 8>

<item 9>
1924 Winter Olympics medal table
wikipedia.org/w/index.php?title=1924_Winter_Olympics_medal_table&oldid=824811466 '' Categories : 1924 Winter Olympics Winter Olympics medal tables Hidden
categories : CS1 French - language sources ( fr ) CS1 Norwegian - language sources ( no ) Use dmy dates from October 2017 Use British English from October 2017
Featured lists Talk Contents About Wikipedia Беларуская Български Català Čeština Deutsch Español Esperanto فارسی Français 한국어 हिन्दी Magyar Македонски
Nederlands 日本 語 Polski Português Русский Slovenščina Suomi Svenska ไทย Türkçe Українська اردو 中文 17 more Edit links This page was last edited on 9 February 2018
, at 17 : 05 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to
the Terms of Use and Privacy Policy .
</item 9>

<item 10>
1924 Winter Olympics medal table
A Norwegian historian discovered that there had been a scoring error , and an American , Anders Haugen , had actually finished in third place . As Haug had died
in 1934 , his daughter presented Haugen , aged 83 , with his medal . In the official report , the classification of nations is ranked not by medals , but rather
by `` points '' , which were awarded for each position from first to sixth ; not exclusively for gold , silver and bronze . Due to this , Czechoslovakia were
included in the table ahead of Belgium , due to two fourth - place finishes , one fifth and one sixth . Italy were also present , in twelfth , for their
solitary sixth place . Contents ( hide ) 1 Medal table 2 See also 3 Notes 4 References Medal table ( edit ) Clas Thunberg won five speed skating medals at the
games for Finland . The medal table is based on information provided by the IOC and is consistent with IOC convention in its published medal tables . By default
, the table is ordered by the number of gold medals the athletes from a nation have won ( in this context , a nation is an entity represented by a National
Olympic Committee ) .
</item 10>
=== END ITEM id=a153-ada74f9430a1b46b ===

=== ITEM id=a175-974d7859b931e445 ===
Gold sentences and facts:
Sentence 0: Cast Character Rank / Position Seasons Notes Regular Recurring Guest Christopher Meloni Elliot Stabler Detective 1 -- 12 Mariska Hargitay Olivia Benson Lieutenant 1 -- Richard Belzer John Munch DA Investigator 1 -- 15 15 , 17 Dann Florek Donald Cragen Captain 1 -- 15 16 Michelle Hurd Monique Jeffries Detective 1 -- 2 Stephanie March Alexandra Cabot Assistant District Attorney 2 -- 5 , 11 2 , 10 , 13 6 Ice - T Odafin `` Fin '' Tutuola Detective 2 -- Tamara Tunie Melinda Warner Medical Examiner 7 -- 12 2 -- 6 , 13 -- 17 19 B.D. Wong George Huang FBI Special Agent 4 -- 12 2 -- 3 13 -- 15 , 17 Diane Neal Casey Novak Senior Assistant District Attorney 5 -- 9 13 12 Adam Beach Chester Lake Detective 9 8 Michaela McManus Kim Greylek Assistant District Attorney 10 Danny Pino Nick Amaro Detective 13 -- 16 Kelli Giddish Amanda Rollins Detective 13 -- Raúl Esparza Rafael Barba Assistant District Attorney 15 -- 14 Peter Scanavino Dominick Carisi Jr . Detective 16 -- 16
  Fact 0.31: Chester Lake is a Detective.

Retrieved items:
<item 1>
Law & Order: Special Victims Unit
Law & Order : Special Victims Unit - Wikipedia Law & Order : Special Victims Unit Jump to : navigation , search For the current season , see Law & Order :
Special Victims Unit ( season 19 ) . Law & Order : Special Victims Unit Also known as Law & Order : SVU SVU Genre Police procedural Legal drama Thriller Created
by Dick Wolf Starring Christopher Meloni Mariska Hargitay Richard Belzer Dann Florek Michelle Hurd Stephanie March Ice - T B.D. Wong Diane Neal Tamara Tunie
Adam Beach Michaela McManus Danny Pino Kelli Giddish Raúl Esparza Peter Scanavino Opening theme Theme of Law & Order : Special Victims Unit Composer ( s ) Mike
Post Country of origin United States Original language ( s ) English No. of seasons 19 No. of episodes 421 ( list of episodes ) Production Executive producer (
s ) Dick Wolf Robert Palm ( season 1 ) David J.
</item 1>

<item 2>
Law & Order: Special Victims Unit
Wong George Huang FBI Special Agent 4 -- 12 2 -- 3 13 -- 15 , 17 Diane Neal Casey Novak Senior Assistant District Attorney 5 -- 9 13 12 Adam Beach Chester Lake
Detective 9 8 Michaela McManus Kim Greylek Assistant District Attorney 10 Danny Pino Nick Amaro Detective 13 -- 16 Kelli Giddish Amanda Rollins Detective 13 --
Raúl Esparza Rafael Barba Assistant District Attorney 15 -- 14 Peter Scanavino Dominick Carisi Jr .
</item 2>

<item 3>
Law & Order: Special Victims Unit
Cast and characters ( edit ) Main article : List of Law & Order : Special Victims Unit characters Cast Character Rank / Position Seasons Notes Regular Recurring
Guest Christopher Meloni Elliot Stabler Detective 1 -- 12 Mariska Hargitay Olivia Benson Lieutenant 1 -- Richard Belzer John Munch DA Investigator 1 -- 15 15 ,
17 Dann Florek Donald Cragen Captain 1 -- 15 16 Michelle Hurd Monique Jeffries Detective 1 -- 2 Stephanie March Alexandra Cabot Assistant District Attorney 2 --
5 , 11 2 , 10 , 13 6 Ice - T Odafin `` Fin '' Tutuola Detective 2 -- Tamara Tunie Melinda Warner Medical Examiner 7 -- 12 2 -- 6 , 13 -- 17 19 B.D.
</item 3>

<item 4>
Law & Order: Special Victims Unit (season 19)
Melinda Warner ) , Carolyn McCormick ( Dr. Elizabeth Olivet ) , and Jenna Stern ( Judge Elana Barth ) would be returning in a January slated episode ( titled ``
Pathological '' ) as their respective characters . Episodes ( edit ) See also : List of Law & Order : Special Victims Unit episodes No . overall No. in season
Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 411 `` Gone Fishin ' '' Alex Chapple Michael Chernuchin September 27 , 2017
( 2017 - 09 - 27 ) 1901 5.67 The Special Victims Unit is notified about a fugitive rapist ( Will Chase ) on the loose in Havana , Cuba . Fin crosses
international borders to capture him , which soon causes political controversy when Fin is accused of kidnapping the perp . When all of his victims are
reluctant to testify , Benson must count on one victim ( Amy Smart ) to help put the perp away .
</item 4>

<item 5>
Law & Order: Special Victims Unit
Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda
Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13
14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' `` Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice
Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion '' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion
'' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and nominations Law & Order franchise Chicago franchise Chicago Fire
Characters Matthew Casey Kelly Severide Episodes Season 1 5 6 Chicago P.D.
</item 5>

<item 6>
Law & Order: Special Victims Unit
Wong , Stephanie March Cast of season 9 ( 2007 -- 08 ) , from left : Diane Neal , Wong , Tamara Tunie , Meloni , Florek , Hargitay , Ice - T , Belzer , Adam
Beach Cast of season 10 ( 2008 -- 09 ) from left : Michaela McManus , Wong , Tunie , Meloni , Hargitay , Florek , Belzer , Ice - T The cast of season 15 ( 2013
-- 2014 ) from left : Raúl Esparza , Florek , Danny Pino , Hargitay , Kelli Giddish , Belzer , Ice - T The cast of seasons 17 -- 18 ( 2015 -- 2017 ) . from left
: Esparza , Giddish , Hargitay , Ice - T , Peter Scanavino Casting for the lead characters of Law & Order : Special Victims Unit occurred in the spring of 1999
. Dick Wolf , along with officials from NBC and Studios USA were at the final auditions for the two leads at Rockefeller Center .
</item 6>

<item 7>
Law & Order: Special Victims Unit (season 19)
Barba ( Raul Esparza ) this season ; Stone 's character is n't necessarily taking his ( Barba 's ) job . `` There 's more than one A.D.A. in New York ( ... )
Philip will come in and he wo n't be the nicest person our characters have ever met . What he 's going to be doing , too , is basically bringing all of the
franchises into one . He 's connecting Law & Order , the Chicago franchises and now SVU . '' Main cast ( edit ) Mariska Hargitay as Lieutenant Olivia Benson
Kelli Giddish as Detective Amanda Rollins Ice - T as Detective Odafin `` Fin '' Tutuola Peter Scanavino as Detective Dominick `` Sonny '' Carisi , Jr . Raúl
Esparza as A.D.A.
</item 7>

<item 8>
Law & Order: Special Victims Unit (season 19)
Missing or empty title = ( help ) ( hide ) Law & Order : Special Victims Unit Characters Detectives Elliot Stabler Olivia Benson John Munch Monique Jeffries Fin
Tutuola Chester Lake Nick Amaro Amanda Rollins Dominick Carisi Jr . Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot
Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya
Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' ``
Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion
'' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion '' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and
nominations Law & Order franchise Retrieved from `` https://en.
</item 8>

<item 9>
Law & Order: Special Victims Unit
In leaving the cast , Florek ended a 400 - episode run as Captain Cragen . In season 16 , Peter Scanavino joined the series , first in a recurring role for
episodes 1 -- 3 and then was promoted to the main cast in episode 5 , with Kelli Giddish , Danny Pino , Ice - T and Raúl Esparza no longer appearing in every
episode . On May 20 , 2015 , it was revealed that Danny Pino would be leaving the cast after the season 16 finale `` Surrendering Noah '' . In August 2017 , it
was announced that Philip Winchester will recur in season 19 as ADA Peter Stone , his character from Chicago P.D. and Chicago Justice , who is the son of
Benjamin Stone , the ADA on the original Law & Order show . It was later also announced that Brooke Shields was enlisted to assume a major recurring role
starting in Season 19 of the long - running dramatic series .
</item 9>

<item 10>
Law & Order: Special Victims Unit
Gillian Hardwicke served as the SVU ADA during season twelve . In season thirteen , both Cabot and Novak returned as ADAs and since the beginning of season
fourteen , ADA Rafael Barba has been SVU 's prosecutor . In season 13 other big changes happened when Stabler left in the season twelve finale . Detectives Nick
Amaro and Amanda Rollins joined the team filling the void left by Stabler . Amaro brings empathy to his cases while dealing with a stressful home life , whilst
Rollins ' dogged persistence and instincts help her close cases , but her secrets could derail her career . In the current seasons the Manhattan SVU is now run
by Lieutenant Olivia Benson . Episodes ( edit ) Main article : List of Law & Order : Special Victims Unit episodes Law & Order : Special Victims Unit has
concluded its eighteenth season and has started its nineteenth season . Each season has aired on NBC and consists of 19 to 25 episodes , each lasting
approximately forty minutes ( sixty minutes including commercials ) . Season Episodes Originally aired Nielsen ratings First aired Last aired Rank Rating 22
September 20 ,
</item 10>
=== END ITEM id=a175-974d7859b931e445 ===
