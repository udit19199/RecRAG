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

=== ITEM id=876f305b27c9a0a9 ===
Gold sentences and facts:
Sentence 0: Cousins Bo ( Seann William Scott ) , Luke ( Johnny Knoxville ) , and Daisy Duke ( Jessica Simpson ) run a moonshine business for their Uncle Jesse ( Willie Nelson ) in Hazzard County , Georgia . The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way , the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) .
  Fact 0.0: Bo, played by Seann William Scott, is a cousin
  Fact 0.1: Luke, played by Johnny Knoxville, is a cousin
  Fact 0.2: Daisy Duke, played by Jessica Simpson, is a cousin
  Fact 0.3: Bo, Luke, and Daisy Duke run a moonshine business
  Fact 0.4: Bo, Luke, and Daisy Duke run the moonshine business for their Uncle Jesse
  Fact 0.5: Uncle Jesse, played by Willie Nelson, is their uncle
  Fact 0.6: The moonshine business is in Hazzard County, Georgia
  Fact 0.7: The cousins' primary mode of transportation is an orange 1969 Dodge Charger
  Fact 0.8: The boys affectionately refer to the orange 1969 Dodge Charger as the 'General Lee'
  Fact 0.9: The family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg
  Fact 0.10: Jefferson Davis Hogg is widely known as 'Boss Hogg'
  Fact 0.11: Boss Hogg is played by Burt Reynolds
  Fact 0.12: Boss Hogg has a willing but dimwitted henchman, Sheriff Rosco P. Coltrane
  Fact 0.13: Sheriff Rosco P. Coltrane is played by M.C. Gainey
  Fact 0.14: Sheriff Rosco P. Coltrane is willing but dimwitted
Sentence 1: Jessica Simpson as Daisy Duke
  Fact 1.0: Jessica Simpson plays Daisy Duke

Retrieved items:
<item 1>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 1>

<item 2>
The Dukes of Hazzard (film)
They are later arrested by Atlanta Police after running from campus police . Back in Hazzard , Daisy learns , with the help of Sheriff 's Deputy Enos Strate (
Michael Weston ) , that Billy Prickett has been hired by Boss Hogg to participate in the rally as a ringer . Boss Hogg then heads to Atlanta , where he informs
the Duke boys , in lock - up , that they are too late to stop him and reveals that the vote on Hogg 's proposition is at the same time as the rally , explaining
Billy Prickett 's involvement . During a transfer from detainment , Daisy helps the boys escape from the patrol car , and they speed home to try to inform the
townsfolk , escaping Atlanta Police , and the Georgia State Patrol after Bo outmaneuvers the city cops . Upon returning home , the Dukes discover that Boss Hogg
and Rosco had taken Uncle Jesse and Pauline hostage , an obvious trap for the boys , and that Billy is in on the scheme because he is ashamed of the town 's low
status .
</item 2>

<item 3>
The Dukes of Hazzard (film)
Jump up ^ `` ' Dukes of Hazzard ' Sheriff Sues Warner Bros. for Millions in Royalties '' . The Hollywood Reporter. 2011 - 08 - 02 . Retrieved 2011 - 12 - 04 .
Jump up ^ Dukes of Hazzard , Cooter 's Place , archived from the original on 2005 - 07 - 16 Jump up ^ NY Daily News , New York , August 5 , 2005 , archived from
the original on 2005 - 10 - 30 , retrieved September 2009 Check date values in : access - date = ( help ) Jump up ^ Exclusive Interview : John Schneider ,
Moviehole , 2007 - 12 - 28 , archived from the original on 2007 - 12 - 28 Jump up ^ `` Jessica Simpson : Singles Chart History '' . billboard.com . Retrieved
2008 - 08 - 07 . Jump up ^ `` USATODAY.com - Jessica Simpson kicks off People 's Choice Awards '' . www.usatoday.com . January 5 , 2006 .
</item 3>

<item 4>
The Dukes of Hazzard (film)
The two race to the farmhouse to cause a distraction to the waiting Hazzard County sheriff 's deputies and Georgia state troopers , while Daisy and Cooter
rescue Jesse and Pauline . Meanwhile , the college girls head to the rally with Sheev to inform the townsfolk about the vote on the strip - mining ordinance .
Because of Sheev 's armadillo hat and lack of pants , no one listens , so Bo leaves for the rally while Luke and Jesse team up to foil the county and state
police who are chasing Bo , interfering with the race . Upon crossing the finish line first , before Billy , the two continue racing back and forth all the way
into town , leading the townsfolk to the courthouse just in time to vote against Boss Hogg 's proposed ordinance . At the courthouse , Daisy takes advantage of
the governor of Georgia 's presence and TV cameras to convince him to pardoning the boys , so Uncle Jesse takes the opportunity to knock out Boss Hogg and gets
a pardon for assaulting a county commissioner at the same time .
</item 4>

<item 5>
The Dukes of Hazzard (film)
Pauline informs the Dukes that Rosco seized another farm on charges , so Bo and Luke investigate a local construction site and find geologic core samples with
the help of bait - shop owner Sheev ( Kevin Heffernan ) . Meanwhile , Coltrane makes arrangements to seize the General Lee as `` evidence '' from the local auto
body shop run by the Dukes ' friend Cooter Davenport ( David Koechner ) , who instead turns the car into a hot rod and applies a new paint job and horn , in
return for finally getting payment for all the work he has done ( `` ... ' cause that 's how this works ... '' ) for the boys in the past . Sheriff 's car at
location in Thousand Oaks , California After retrieving the General Lee before Rosco can , the Dukes go to Atlanta to visit a local university geology lab ,
meeting with Katie - Lynn Johnson ( Nikki Griffin ) , a Hazzard county girl and the Dukes ' love interest , and her Australian roommate Annette ( Jacqui Maxwell
) . At the lab , they discover Boss Hogg 's intentions of turning the county into a strip coal mine .
</item 5>

<item 6>
The Dukes of Hazzard (film)
The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way ,
the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but
dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) . After Rosco has the General Lee impounded after Bo and Luke 's attempt to run away from a
daughter of one their many moonshine customers , Billy Prickett ( James Roday ) , a famous stock - car driver , enters Hazzard to participate in the rally .
Meanwhile , Rosco plants a fake moonshine still ( `` ' cause he 's too dumb to find the real one '' ) in Uncle Jesse 's barn and seizes the Duke property in the
interest of eminent domain for Boss Hogg , forcing the family to temporarily reside with neighbor and Uncle Jesse 's love - interest , Pauline ( Lynda Carter )
.
</item 6>

<item 7>
The Dukes of Hazzard (film)
The final scene shows a cook - out at the Dukes ' house where Pauline convinces Uncle Jesse , who could not be found because he was `` using the meat smoker ''
, to get up and play the television series ' main theme . Bo and Luke are romantically involved with the girls in the General Lee when they are caught by Luke
's other love - interest Laurie Pullman ( Alice Greczyn ) from the introduction of the film , who proceeds to chase them with a shotgun as they drive away .
During the credits , outtakes from the production of the film are shown . Cast ( edit ) Knoxville and Scott at the premiere Johnny Knoxville as Luke Duke Seann
William Scott as Bo Duke Jessica Simpson as Daisy Duke Burt Reynolds as Boss Hogg Willie Nelson as Uncle Jesse Duke David Koechner as Cooter Davenport M.C.
</item 7>

<item 8>
The Dukes of Hazzard (film)
The Dukes of Hazzard ( film ) - wikipedia The Dukes of Hazzard ( film ) Jump to : navigation , search The Dukes of Hazzard Theatrical release poster Directed by
Jay Chandrasekhar Produced by Bill Gerber Screenplay by John O'Brien Story by John O'Brien Jonathan L. Davis Based on Characters by Gy Waldron Starring Johnny
Knoxville Seann William Scott Jessica Simpson Burt Reynolds Willie Nelson Narrated by Junior Brown Music by Nathan Barr Cinematography Lawrence Sher Edited by
Lee Haxall Myron I. Kerstein Production company Village Roadshow Pictures Distributed by Warner Bros. Pictures Release date August 5 , 2005 ( 2005 - 08 - 05 )
Running time 104 minutes Country United States Language English Budget $50 million Box office $111.1 million The Dukes of Hazzard is a 2005 American buddy
comedy road film based on the television series , The Dukes of Hazzard . The film was directed by Jay Chandrasekhar and released on August 5 , 2005 , by Warner
Bros. Pictures .
</item 8>

<item 9>
The Dukes of Hazzard (film)
Pictures . As in the television series , the film depicts the adventures of cousins Bo , Luke , and Daisy , and their Uncle Jesse , as they outfox crooked
Hazzard County Commissioner Boss Hogg and Sheriff Rosco P. Coltrane . The film was the acting debut of pop singer Jessica Simpson . While financially successful
, the film met with negative reviews from critics . The film was followed by a direct - to - video prequel titled The Dukes of Hazzard : The Beginning ( 2007 )
. Contents ( hide ) 1 Plot 2 Cast 2.1 Cameos 3 Production 3.1 Stunts 3.2 Locations 4 Reception 4.1 Box office 4.2 Critical reception 4.3 Awards 5 Controversy 6
Soundtrack 7 Unrated Cut 8 Prequel 9 References 10 External links Plot ( edit ) Cousins Bo ( Seann William Scott ) , Luke ( Johnny Knoxville ) , and Daisy Duke
( Jessica Simpson ) run a moonshine business for their Uncle Jesse ( Willie Nelson ) in Hazzard County , Georgia .
</item 9>

<item 10>
The Dukes of Hazzard (film)
In a film review , a New York Daily News entertainment columnist said the movie 's sex humor is `` cruder '' than the TV series , but that it is `` nearly
identical to the TV series in ... its ogling of the posterior of cousin Daisy Duke . '' Although initially he commented that he enjoyed the new style of
relationship between the movie versions of Bo and Luke , John Schneider , who played Bo Duke in the original series , was later asked if he saw the film and
said : `` My gosh ... it was terrible ! It was n't Dukes . It was true to whatever it was ; I just do n't know what that was ! '' Soundtrack ( edit ) Main
article : These Boots Are Made for Walkin ' § Jessica Simpson version Jessica Simpson recorded her own version of `` These Boots Are Made for Walkin ' '' ( and
added her own lyrics ) for the soundtrack to the film The Dukes of Hazzard ( 2005 ) . Performed from the point of view of her character in the movie , Simpson
's cover was co-produced by Jimmy Jam and Terry Lewis , and was released as the soundtrack 's first single in 2005 .
</item 10>
=== END ITEM id=876f305b27c9a0a9 ===

=== ITEM id=21872926883d4a8c ===
Gold sentences and facts:
Sentence 0: No . Name Province Birth / death Term of active service Term of service as Chief Length in days of active service Length in days as Chief Appointed by Law school Position prior to appointment Codrington Edmund Carrington England 1769 - 1849 March 1801 -- 2 April 1806 March 1801 -- 2 April 1806 North -- Counsel to the East India Company Edmund Henry Lushington England 1766 - 1839 5 September 1801 -- 1809 15 April 1807 - 1809 North University of Cambridge Barrister Alexander Johnston Scotland 1775 - 1849 15 April 1807 -- 14 February 1809 6 November 1811 - 1819 6 November 1811 - 1819 671 + ? Maitland University of Göttingen Advocate Fiscal of Ceylon William Coke England 1776 - 1818 6 March 1809 -- 1 September 1818 † 6 March 1809 -- 1811 3,466 Maitland University of Oxford Advocate Fiscal of Ceylon 5 Henry Byrne 5 November 1818 -- 1819 -- -- Brownrigg Barrister 6 Ambrose Hardinge Giffard Ireland 1771 -- 1827 8 April 1819 -- 2 March 1827 8 April 1819 -- 2 March 1827 2,885 2,885 Brownrigg Trinity College , Dublin Advocate Fiscal of Ceylon 7 Richard Ottley 1782 -- 1845 8 May 1820 -- 1833 1 November 1827 -- 1833 Barnes Chief Justice of Grenada 8 Henry Matthews England 1789 -- 1828 10 October 1827 -- 1828 -- -- Barnes King 's College , Cambridge Advocate Fiscal of Ceylon 9 Charles Marshall England 1788 - 1873 14 June 1828 -- 3 March 1836 18 February 1833 -- 3 March 1836 2,850 1,109 Barnes Jesus College , Cambridge 10 William Rough England 1772 - 1838 18 February 1833 - 1838 -- -- Trinity College , Cambridge 11 William Norris 1834 -- 1837 27 April 1836 -- 1837 Wilmot - Horton Advocate Fiscal of Ceylon 12 John Jeremie Guernsey 1795 - 1841 10 March 1836 -- 1840 -- -- Wilmot - Horton Public prosecutor of Mauritius 13 John Fredrick Stoddart Malta 1805 - 1839 2 June 1838 -- 19 August 1839 † -- 443 -- University of Glasgow 14 Anthony Oliphant Scotland 1793 -- 1859 22 October 1838 - 1854 22 October 1838 - 1854 Attorney General of the Cape Colony 15 J.G. Hildebrand 23 November 1839 -- -- -- 16 William Ogle Carr England 1802 - 1856 19 December 1939 - 1856 17 April 1854 -- 1856 Stewart - Mackenzie King 's Advocate of Ceylon 17 John Stark 10 October 1840 -- -- -- Queen 's Advocate of Ceylon 18 Christopher Temple 28 January 1854 -- -- -- Judge of the District Court of Colombo 19 Paul Ivy Sterling 20 November 1855 -- -- -- Attorney General of Hong Kong 20 William Carpenter Rowe - 1860 1857 -- 1859 -- 21 Edward Shepherd Creasy England 1812 - 1878 -- 27 March 1860 - 1875 University of Cambridge 22 Richard Morgan -- -- -- Queen 's Advocate of Ceylon 23 George Lawson -- -- -- 24 Henry Byerley Thompson - 1867 1 January 1863 -- -- -- Queen 's Advocate of Ceylon 25 Charles Henry Stewart 11 January 1867 -- -- -- 26 Richard Cayley England 26 July 1874 -- 1882 1 October 1879 -- 1882 -- Queen 's Advocate of Ceylon 27 Lovell Burchett Clarence 5 July 1879 -- -- -- 28 Harry Dias Bandaranaike 5 July 1879 -- -- -- 29 William Hackett 3 February 1877 - 1877 3 February 1877 - 1877 -- 30 John Budd Phear England 18 October 1877 -- -- -- 31 Archibald Campbell Lawrie 1837 - 1914 6 July 1892 -- 1901 -- -- Judge of the District Court of Kandy 32 Jacobus de Wet South Africa 31 May 1882 -- 29 May 1883 31 May 1882 -- 29 May 1883 363 363 33 Bruce Burnside Bahamas 21 May 1883 -- 1893 21 May 1883 -- 1893 -- Queen 's Advocate of Ceylon 34 John Winfield Bonser 13 November 1893 -- 1902 13 November 1893 -- 1902 Chief Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40 Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42 Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,651 1,651 45 James Cecil Walter Pereira 4 July 1912 -- -- -- 46 James Arthur Van Langenburg -- -- -- 47 Thomas De Sampayo 1915 -- -- -- University of Cambridge Commissioner of Assize 48 George Francis Macdaniel Ennis 5 August 1912 -- -- -- 49 Walter Shaw 10 December 1914 -- -- -- Chief Justice of British Honduras 50 Gualterus Stewart Schneider -- -- -- 51 Anton Bertram 1869 - 1937 26 July 1918 -- 1925 26 July 1918 -- 1925 University of Cambridge Attorney General of Ceylon 52 Hermann Albert Loos -- -- -- 53 William Thomas Porter 12 January 1922 -- -- -- 54 Thomas Garvin 1881 - 1940 21 June 1921 -- 1935 -- -- Manning Colombo Law College Solicitor General of Ceylon 55 John Adrian St. Valentine Jayewardene 1877 -- 1927 1923 -- -- -- Manning Judge of the District Court of Colombo 56 Llwellyn Chisholm Dalton 14 March 1925 -- -- -- 57 Lewis Matthew Maartensz 18 February 1935 -- -- -- 58 Maas Thajoon Akbar 1880 - 1944 -- -- -- University of Cambridge Solicitor General of Ceylon 59 Charles Ernest St. John Branch 1865 - 1939 3 July 1925 -- 25 May 1926 3 July 1925 -- 25 May 1926 326 326 Durham University 60 Stanley Fisher 1867 - 11 December 1926 -- 1930 11 December 1926 -- 1930 Chief Justice of Trinidad and Tobago 61 Robert William Lyall Grant 17 February 1926 -- -- -- 62 Allen Drieberg 19 September 1927 -- -- -- 63 Eugene Wilfred Jayewardene Western 1874 - 1932 May 1928 -- -- -- Fletcher Member of the Colombo Municipal Council 64 Philip James Macdonell 1873 - 1940 3 October 1930 - 1936 3 October 1930 - 1936 Chief Justice of Trinidad and Tobago 65 Lucian Macull Dominic de Silva -- -- -- 66 William Edward Barber -- -- -- 67 Francis Harold Bertram 19 February 1935 -- -- -- 68 Kenneth Poyser -- -- -- 69 Francis Soertsz - 1951 12 November 1936 -- -- -- 70 Sidney Abrahams England 1885 - 1957 3 July 1936 -- 1939 3 July 1936 -- 1939 University of Cambridge 71 Francis Moseley 2 August 1936 -- -- -- 72 Vincent Fernando 11 November 1937 -- -- -- 73 Hector Hearne 21 May 1937 -- -- -- 74 Arthur Eric Keuneman 4 March 1938 -- -- -- University of Cambridge 75 Oswald Leslie De Kretser II Western 1882 - 1959 8 April 1938 -- 1945 -- -- Caldecott University College , Colombo Commissioner of Assize 76 Arthur Wijewardena 1887 - 1964 22 August 1938 -- 1950 15 January 1949 -- 1950 University of Cambridge Solicitor General of Ceylon 77 J.H.B. Nihill 18 November 1938 -- -- -- 78 John Curtois Howard 1 December 1939 -- 1945 1 December 1939 -- 1945 Attorney General of Cyprus 79 G.H.F. Cannon -- -- -- 80 Edward Jayetileke 14 May 1942 -- 11 October 1951 1950 -- 11 October 1951 3,437 Caldecott Solicitor General of Ceylon 81 Alan Rose 11 January 1945 -- 1956 11 October 1951 -- 1956 82 A.R.H. Canakeratne 1886 - 1960 28 March 1945 -- -- -- Ceylon Law College University of London Commissioner of Assize 83 Manikku Wadumestri Hendrick de Silva -- -- -- Attorney General of Ceylon 84 Felix Reginald Dias Bandaranaike I 1861 - 1947 3 September 1946 -- -- -- University of Cambridge Judge of the District Court of Colombo 85 Chellappah Nagalingam 1893 - 1958 22 July 1947 -- 1958 -- -- Ceylon Law College Attorney General of Ceylon 86 Ralph Windham 23 July 1947 -- -- -- 87 Hema Henry Basnayake 1902 - 23 October 1947 -- 3 August 1964 1 January 1956 -- 3 August 1964 6,129 3,137 Senanayake Attorney General of Ceylon 88 Edward Frederick Noel Gratiaen 22 March 1948 -- -- -- Senanayake Attorney General of Ceylon 89 E.H.T. Gunasekera 1 March 1949 -- 25 January 1962 -- 4,713 -- Senanayake 90 M.F.S. Pulle 28 September 1949 -- -- -- Senanayake 91 V.L. St. C. Swan 1 May 1950 -- -- -- Senanayake 92 H.A. de Silva 30 November 1950 -- -- -- Senanayake 93 Kairshasp Choksy Western -- -- -- 94 Henry Weerasooriya 10 July 1953 -- 24 February 1964 -- 3,881 -- Senanayake 95 K.D. de Silva 10 July 1953 -- -- -- Senanayake 96 Miliani Sansoni 10 May 1954 -- 17 November 1966 3 August 1964 -- 17 November 1966 4,574 836 Kotelawala 97 Hugh Fernando 1910 - 1976 16 May 1955 -- 17 November 1973 20 November 1966 - 17 November 1973 6,760 2,554 Kotelawala 98 Thusew Samuel Fernando 2 May 1956 -- -- -- Bandaranaike Colombo Law College Solicitor General of Sri Lanka 99 Nadaraja Sinnetamby 3 May 1956 -- -- -- Bandaranaike 100 Lionel de Silva 5 November 1959 -- -- -- Dahanayake 101 Leonard de Silva 16 August 1966 -- -- -- Senanayake 102 Henry Thambiah Northern 1906 -- 1997 1 December 1961 -- 27 May 1968 -- 2,369 -- Bandaranaike University College , Colombo Ceylon Law College University of London Commissioner of Assize 103 Kingsley Herath 25 January 1962 -- -- -- Bandaranaike 104 Asoka Abeyesundere Southern 1906 - 1988 20 April 1962 -- 3 May 1968 -- 2,205 -- Bandaranaike Ceylon Law College 105 Ponnuduraiswamy Sriskandarajah 1905 - 1968 22 June 1962 -- 8 February 1967 -- 1,692 -- Bandaranaike Ceylon University College Ceylon Law College Commissioner of Assize 106 Gardiye Punchihewage Amaraseela Silva 22 June 1962 -- 1974 1973 -- 1974 -- Bandaranaike 107 Lionel Sirimanne -- -- -- Bandaranaike 108 Vaithalingam Manicavasagar Western 1906 - 1993 17 December 1964 -- 14 December 1967 -- 1,092 -- Bandaranaike Colombo Law College Commissioner of Assize 109 Anthony Alles Central 1911 -- 2003 1964 -- 1974 -- -- Bandaranaike University of Ceylon Ceylon Law College University of London Solicitor General of Sri Lanka 110 Veeravagu Sivasubramaniam 1908 - 1985 1966 -- 1970 -- -- Senanayake University of London Commissioner of Assize 111 George Samerawickrame 18 November 1966 -- 9 April 1981 -- 5,256 -- Senanayake 112 Victor Tennekoon Central 1914 - 8 February 1967 -- 1 July 1970 1 January 1974 - 8 September 1977 1 January 1974 - 8 September 1977 1,239 + 1,346 1,346 Senanayake University of London Ceylon Law College Judge of the Court of Appeal 113 Christopher Weeramantry Western 1926 - 1967 -- 30 April 1972 -- -- Senanayake University of Ceylon University of London Commissioner of Assize 114 Oswald Leslie De Kretser III Western 1910 - 1968 -- 21 January 1972 -- -- Senanayake University College , Colombo University of London Judge of the District Court of Panadura 115 Samarapilimudalige Wijayatilake Central 1912 - 1990 1968 -- 16 May 1975 -- -- Senanayake University College , Colombo Ceylon Law College Commissioner of Assize 116 Pandita Gunawadena 1968 -- 10 February 1971 -- -- Senanayake 117 Vincent Thamotheram 1915 - 23 March 1970 -- 1980 -- -- Senanayake Ceylon University College Ceylon Law College Commissioner of Assize 118 Hector Deheragoda 22 February 1972 -- 27 December 1976 -- 1,770 -- Bandaranaike 119 Cuda Banda Walgampaya 22 February 1972 -- 3 January 1976 -- 1,411 -- Bandaranaike 120 Jaya Pathirana - 2000 22 February 1972 -- 1978 -- -- Bandaranaike 121 Dharmadasa Wimalaratne 10 April 1972 - ? 20 October 1980 -- 15 September 1985 -- 1,791 -- Bandaranaike 122 Tellipalai Rajaratnam 1920 - 1994 10 April 1972 -- 1978 -- -- Bandaranaike Commissioner of Assize 123 Donald Sirimanne 1973 -- -- -- Gopallawa 124 Cyril Udalagama 1 January 1974 -- -- -- Gopallawa 125 T.A. de S. Wijesundera 1 January 1974 -- -- -- Gopallawa 126 Malcolm Perera 1 January 1974 -- -- -- Gopallawa 127 Izadean Mohamed Ismail 1 January 1974 -- 18 December 1981 -- 2,908 -- Gopallawa 128 Joe Weereratne 1 January 1974 -- -- -- Gopallawa 129 Arulampalam Vythialingam 1 January 1974 -- -- -- Gopallawa 130 Noel Tittawela 1 January 1974 -- -- -- Gopallawa 131 Suppiah Sharvananda Northern 1923 - 2007 1 January 1974 -- 1988 29 October 1984 -- 1988 Gopallawa Colombo Law College University of London 132 Somaratne Walpita 1 January 1974 -- -- -- Gopallawa 133 Wilmot Gunasekera 2 May 1974 -- -- -- Gopallawa 134 Barnes Ratwatte II 1918 - 2004 2 December 1974 -- ? 11 April 1981 -- 22 November 1983 -- 955 -- Gopallawa Jayewardene Ceylon Law College President of the Court of Appeal of Sri Lanka 135 Raja Wanasundera Sabaragamuwa 1924 - 2012 5 June 1975 -- ? -- -- Gopallawa University College , Colombo Ceylon Law College University of London Stanford University Solicitor General of Sri Lanka 136 Percy Colin - Thomé Southern 1922 - 12 January 1976 -- 1977 29 December 1981 -- ? -- -- Gopallawa Jayewardene University of Ceylon University of Cambridge Judge of the High Court of Colombo 137 Neville Samarakoon 1919 - 1990 -- 21 October 1984 1977 -- 21 October 1984 Jayewardene University College , Colombo Colombo Law College 138 Victor Perera 24 July 1981 -- 18 August 1983 -- 755 -- Jayewardene 139 Anton Soza Western 1919 - 2013 26 January 1982 -- 3 February 1984 -- 738 -- Jayewardene University of London Judge of the Court of Appeal 140 Parinda Ranasinghe 15 November 1982 -- 1991 1988 -- 1991 Jayewardene Colombo Law College 141 Abdul Cader 8 December 1982 -- 28 May 1985 -- 902 -- Jayewardene 142 Heevi Rodrigo 8 December 1982 -- 4 December 1984 -- 727 -- Jayewardene 143 E.A.D. Atukorale 27 November 1984 -- -- -- Jayewardene 144 Herbert Thambiah Northern 1926 -- 1992 27 November 1984 -- 1992 1991 -- 1991 Jayewardene University of Ceylon Ceylon Law College Judge of the Court of Appeal 145 Lucian Hector de Alwis 27 November 1984 -- -- -- Jayewardene 146 O.S.M. Seneviratne 10 February 1986 -- -- -- Jayewardene 147 H.A.G. de Silva 10 February 1986 -- -- -- Jayewardene Mark Fernando 1941 - 2009 4 March 1988 -- 2005 -- -- Jayewardene University of Ceylon , Peradeniya A.R.B. Amerasinghe 1937 - 2013 -- -- -- A.R.N. Fernando -- -- -- Ameer Ismail 1990 -- 1994 -- -- Premadasa University of Ceylon Ceylon Law College President of the Court of Appeal of Sri Lanka Andrew Ranjan Perera -- -- -- Pathmanathan Ramanathan 1932 - 2006 -- -- -- University of Wales President of the Court of Appeal of Sri Lanka K.M.M.B. Kulatunge -- -- -- G.P.S. de Silva -- 1991 -- 1999 Sarath N. Silva 1995 -- 16 September 1999 16 September 1999 -- 7 June 2009 3,552 Kumaratunga Sri Lanka Law College University of Brussels President of the Court of Appeal of Sri Lanka Shirani Bandaranayake North Western 1958 - 30 October 1996 -- 13 January 2013 28 January 2015 -- 29 January 2015 18 May 2011 -- 13 January 2013 28 January 2015 -- 29 January 2015 5,919 + 1 606 + 1 Kumaratunga University of Colombo University of London Lawyer C.V. Vigneswaran Western 1939 - March 2001 -- October 2004 -- -- Kumaratunga University of Ceylon Ceylon Law College Judge of the Court of Appeal Asoka de Silva 1946 - 1 August 2001 -- 17 May 2011 8 June 2009 -- 17 May 2011 3,576 708 Kumaratunga University of Peradeniya University of Colombo University of Illinois at Urbana -- Champaign President of the Court of Appeal of Sri Lanka Nihal Jayasinghe 19 February 2003 -- 29 February 2008 -- 1,836 -- Kumaratunga University of Colombo Judge of the Court of Appeal Nimal Dissanayake -- 23 April 2008 -- -- Kumaratunga Shiranee Tilakawardane 19 February 2003 - 5 April 2014 -- 1,901 -- Kumaratunga Sri Lanka Law College President of the Court of Appeal of Sri Lanka Nimal Gamini Amaratunga January 2005 -- March 2014 -- -- Kumaratunga University of Colombo International Maritime Law Institute Judge of the Court of Appeal Saleem Marsoof February 2005 -- 2014 -- -- Kumaratunga University of Colombo , University of San Diego School of Law President of the Court of Appeal of Sri Lanka Andrew Somawansa North Western 28 May 2006 -- October 2008 -- -- Rajapaksa University of Ceylon University of Nevada President of the Court of Appeal of Sri Lanka Jagath Balapatabendi Southern -- -- -- Rajapaksa Ceylon Law College Judge of the Court of Appeal Kanagasabapathy Sripavan † 1952 -- 27 March 2008 -- present 30 January 2015 -- 28 February 2017 7003368200000000000 ♠ 3,682 7003118200000000000 ♠ 1,182 Rajapaksa Sri Lanka Law College Judge of the Court of Appeal P.A. Ratnayake 2 May 2008 -- 15 February 2013 -- 7003175100000000000 ♠ 1,751 -- Rajapaksa University of Brussels Chandra Ekanayake November 2008 -- 2016 -- -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka S.I. Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 -- present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,119 -- Rajapaksa Sri Lanka Law College University of Leicester Attorney General of Sri Lanka Mohan Peiris ♠ 15 January 2013 -- 28 January 2015 15 January 2013 -- 28 January 2015 743 743 Rajapaksa Sri Lanka Law College Attorney General of Sri Lanka Rohini Marasinghe 26 April 2013 -- 2015 -- -- Rajapaksa University of Hartford Judge of the Court of Appeal Buwaneka Aluwihare † 4 December 2013 -- present -- 7003160400000000000 ♠ 1,604 -- Rajapaksa University of London Additional Solicitor General of Sri Lanka Sisira de Abrew † Western 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne † 30 January 2015 -- present -- 7003118200000000000 ♠ 1,182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present -- 7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 -- present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠ 679 -- Sirisena University of Colombo Judge of the Court of Appeal
  Fact 0.0: Edmund Carrington was from England, born 1769 and died 1849.
  Fact 0.1: Edmund Carrington's active service term was March 1801 to 2 April 1806.
  Fact 0.2: Edmund Carrington's province was North.
  Fact 0.3: Edmund Carrington's prior position was counsel to the East India Company.
  Fact 0.4: Edmund Henry Lushington was from England, born 1766 and died 1839.
  Fact 0.5: Edmund Henry Lushington's active service term began 5 September 1801 and ended in 1809.
  Fact 0.6: Edmund Henry Lushington's term as chief was 15 April 1807 to 1809.
  Fact 0.7: Edmund Henry Lushington's province was North.
  Fact 0.8: Edmund Henry Lushington studied at the University of Cambridge.
  Fact 0.9: Edmund Henry Lushington's prior position was barrister.
  Fact 0.10: Alexander Johnston was from Scotland, born 1775 and died 1849.
  Fact 0.11: Alexander Johnston's active service term was 15 April 1807 to 14 February 1809.
  Fact 0.12: Alexander Johnston's term as chief was 6 November 1811 to 1819.
  Fact 0.13: Alexander Johnston was appointed by Maitland.
  Fact 0.14: Alexander Johnston studied at the University of Göttingen.
  Fact 0.15: Alexander Johnston's prior position was Advocate Fiscal of Ceylon.
  Fact 0.16: William Coke was from England, born 1776 and died 1818.
  Fact 0.17: William Coke's active service term was 6 March 1809 to 1 September 1818.
  Fact 0.18: William Coke's term as chief was 6 March 1809 to 1811.
  Fact 0.19: William Coke was appointed by Maitland.
  Fact 0.20: William Coke studied at the University of Oxford.
  Fact 0.21: William Coke's prior position was Advocate Fiscal of Ceylon.
  Fact 0.22: Henry Byrne was chief from 5 November 1818 to 1819.
  Fact 0.23: Henry Byrne was appointed by Brownrigg.
  Fact 0.24: Henry Byrne's prior position was barrister.
  Fact 0.25: Ambrose Hardinge Giffard was from Ireland, born 1771 and died 1827.
  Fact 0.26: Ambrose Hardinge Giffard's active service term was 8 April 1819 to 2 March 1827.
  Fact 0.27: Ambrose Hardinge Giffard was appointed by Brownrigg.
  Fact 0.28: Ambrose Hardinge Giffard studied at Trinity College, Dublin.
  Fact 0.29: Ambrose Hardinge Giffard's prior position was Advocate Fiscal of Ceylon.
  Fact 0.30: Richard Ottley lived from 1782 to 1845.
  Fact 0.31: Richard Ottley's active service term was 8 May 1820 to 1833.
  Fact 0.32: Richard Ottley's term as chief was 1 November 1827 to 1833.
  Fact 0.33: Richard Ottley was appointed by Barnes.
  Fact 0.34: Richard Ottley's prior position was Chief Justice of Grenada.
  Fact 0.35: Henry Matthews was from England, born 1789 and died 1828.
  Fact 0.36: Henry Matthews's active service term was 10 October 1827 to 1828.
  Fact 0.37: Henry Matthews was appointed by Barnes.
  Fact 0.38: Henry Matthews studied at King's College, Cambridge.
  Fact 0.39: Henry Matthews's prior position was Advocate Fiscal of Ceylon.
  Fact 0.40: Charles Marshall was from England, born 1788 and died 1873.
  Fact 0.41: Charles Marshall's active service term was 14 June 1828 to 3 March 1836.
  Fact 0.42: Charles Marshall's term as chief was 18 February 1833 to 3 March 1836.
  Fact 0.43: Charles Marshall was appointed by Barnes.
  Fact 0.44: Charles Marshall studied at Jesus College, Cambridge.
  Fact 0.45: William Rough was from England, born 1772 and died 1838.
  Fact 0.46: William Rough's term was 18 February 1833 to 1838.
  Fact 0.47: William Rough studied at Trinity College, Cambridge.
  Fact 0.48: William Norris's term was 27 April 1836 to 1837.
  Fact 0.49: William Norris was appointed by Wilmot-Horton.
  Fact 0.50: William Norris's prior position was Advocate Fiscal of Ceylon.
  Fact 0.51: John Jeremie was from Guernsey, born 1795 and died 1841.
  Fact 0.52: John Jeremie's term was 10 March 1836 to 1840.
  Fact 0.53: John Jeremie was appointed by Wilmot-Horton.
  Fact 0.54: John Jeremie's prior position was public prosecutor of Mauritius.
  Fact 0.55: John Fredrick Stoddart was from Malta, born 1805 and died 1839.
  Fact 0.56: John Fredrick Stoddart's active service term was 2 June 1838 to 19 August 1839.
  Fact 0.57: John Fredrick Stoddart's length of active service was 443 days.
  Fact 0.58: John Fredrick Stoddart studied at the University of Glasgow.
  Fact 0.59: Anthony Oliphant was from Scotland, born 1793 and died 1859.
  Fact 0.60: Anthony Oliphant's active service term was 22 October 1838 to 1854.
  Fact 0.61: Anthony Oliphant's term as chief was 22 October 1838 to 1854.
  Fact 0.62: Anthony Oliphant's prior position was Attorney General of the Cape Colony.
  Fact 0.63: J.G. Hildebrand's term began 23 November 1839.
  Fact 0.64: William Ogle Carr was from England, born 1802 and died 1856.
  Fact 0.65: William Ogle Carr's term was 17 April 1854 to 1856.
  Fact 0.66: William Ogle Carr was appointed by Stewart-Mackenzie.
  Fact 0.67: William Ogle Carr's prior position was King's Advocate of Ceylon.
  Fact 0.68: John Stark's term began 10 October 1840.
  Fact 0.69: John Stark's prior position was Queen's Advocate of Ceylon.
  Fact 0.70: Christopher Temple's term began 28 January 1854.
  Fact 0.71: Christopher Temple's prior position was judge of the District Court of Colombo.
  Fact 0.72: Paul Ivy Sterling's term began 20 November 1855.
  Fact 0.73: Paul Ivy Sterling's prior position was Attorney General of Hong Kong.
  Fact 0.74: William Carpenter Rowe died in 1860.
  Fact 0.75: William Carpenter Rowe's term was 1857 to 1859.
  Fact 0.76: Edward Shepherd Creasy was from England, born 1812 and died 1878.
  Fact 0.77: Edward Shepherd Creasy's term was 27 March 1860 to 1875.
  Fact 0.78: Edward Shepherd Creasy studied at the University of Cambridge.
  Fact 0.79: Richard Morgan's prior position was Queen's Advocate of Ceylon.
  Fact 0.80: George Lawson has no recorded dates in the table.
  Fact 0.81: Henry Byerley Thompson died in 1867.
  Fact 0.82: Henry Byerley Thompson's term began 1 January 1863.
  Fact 0.83: Henry Byerley Thompson's prior position was Queen's Advocate of Ceylon.
  Fact 0.84: Charles Henry Stewart's term began 11 January 1867.
  Fact 0.85: Richard Cayley was from England.
  Fact 0.86: Richard Cayley's term was 26 July 1874 to 1882.
  Fact 0.87: Richard Cayley's term as chief was 1 October 1879 to 1882.
  Fact 0.88: Richard Cayley's prior position was Queen's Advocate of Ceylon.
  Fact 0.89: Lovell Burchett Clarence's term began 5 July 1879.
  Fact 0.90: Harry Dias Bandaranaike's term began 5 July 1879.
  Fact 0.91: William Hackett's term was 3 February 1877 to 1877.
  Fact 0.92: John Budd Phear was from England and his term began 18 October 1877.
  Fact 0.93: Archibald Campbell Lawrie lived from 1837 to 1914.
  Fact 0.94: Archibald Campbell Lawrie's term was 6 July 1892 to 1901.
  Fact 0.95: Archibald Campbell Lawrie's prior position was judge of the District Court of Kandy.
  Fact 0.96: Jacobus de Wet was from South Africa.
  Fact 0.97: Jacobus de Wet's term was 31 May 1882 to 29 May 1883, a length of 363 days.
  Fact 0.98: Bruce Burnside was from the Bahamas.
  Fact 0.99: Bruce Burnside's term was 21 May 1883 to 1893.
  Fact 0.100: Bruce Burnside's prior position was Queen's Advocate of Ceylon.
  Fact 0.101: John Winfield Bonser's term was 13 November 1893 to 1902.
  Fact 0.102: John Winfield Bonser's prior position was Chief Justice of the Straits Settlements.
  Fact 0.103: George Henry Withers has no recorded dates in the table.
  Fact 0.104: Dodwell Browne has no recorded dates in the table.
  Fact 0.105: Fredrick Charles Moncrieff's term began 4 April 1900.
  Fact 0.106: Henry Lorentz Wendt's term was 15 October 1901 to 28 November 1910.
  Fact 0.107: John Page Middleton's term was 28 January 1902 to 1912.
  Fact 0.108: John Page Middleton's prior position was judge of the Supreme Court of Cyprus.
  Fact 0.109: Charles Layard Western's term was 26 April 1902 to 18 June 1906, a length of 1,514 days.
  Fact 0.110: Joseph Richard Grenier's term was 28 November 1910 to 1912.
  Fact 0.111: Alexander Wood Renton's term was 3 September 1905 to 1918.
  Fact 0.112: Alexander Wood Renton's term as chief was 22 August 1914 to 1918.
  Fact 0.113: Alfred George Lascelles was from England and his term was 1 May 1911 to 1914.
  Fact 0.114: Joseph Turner Hutchinson was from England and his term was 23 October 1906 to 1 May 1911, a length of 1,651 days.
  Fact 0.115: James Cecil Walter Pereira's term began 4 July 1912.
  Fact 0.116: James Arthur Van Langenburg has no recorded dates in the table.
  Fact 0.117: Thomas De Sampayo's term began in 1915.
  Fact 0.118: Thomas De Sampayo studied at the University of Cambridge.
  Fact 0.119: Thomas De Sampayo's prior position was Commissioner of Assize.
  Fact 0.120: George Francis Macdaniel Ennis's term began 5 August 1912.
  Fact 0.121: Walter Shaw's term began 10 December 1914.
  Fact 0.122: Walter Shaw's prior position was Chief Justice of British Honduras.
  Fact 0.123: Gualterus Stewart Schneider has no recorded dates in the table.
  Fact 0.124: Anton Bertram lived from 1869 to 1937.
  Fact 0.125: Anton Bertram's term was 26 July 1918 to 1925.
  Fact 0.126: Anton Bertram studied at the University of Cambridge.
  Fact 0.127: Anton Bertram's prior position was Attorney General of Ceylon.
  Fact 0.128: Hermann Albert Loos has no recorded dates in the table.
  Fact 0.129: William Thomas Porter's term began 12 January 1922.
  Fact 0.130: Thomas Garvin lived from 1881 to 1940.
  Fact 0.131: Thomas Garvin's term was 21 June 1921 to 1935.
  Fact 0.132: Thomas Garvin studied at Colombo Law College.
  Fact 0.133: Thomas Garvin's prior position was Solicitor General of Ceylon.
  Fact 0.134: John Adrian St. Valentine Jayewardene lived from 1877 to 1927.
  Fact 0.135: John Adrian St. Valentine Jayewardene's term began in 1923.
  Fact 0.136: John Adrian St. Valentine Jayewardene was appointed by Manning.
  Fact 0.137: John Adrian St. Valentine Jayewardene's prior position was judge of the District Court of Colombo.
  Fact 0.138: Llwellyn Chisholm Dalton's term began 14 March 1925.
  Fact 0.139: Lewis Matthew Maartensz's term began 18 February 1935.
  Fact 0.140: Maas Thajoon Akbar lived from 1880 to 1944.
  Fact 0.141: Maas Thajoon Akbar studied at the University of Cambridge.
  Fact 0.142: Maas Thajoon Akbar's prior position was Solicitor General of Ceylon.
  Fact 0.143: Charles Ernest St. John Branch lived from 1865 to 1939.
  Fact 0.144: Charles Ernest St. John Branch's term was 3 July 1925 to 25 May 1926, a length of 326 days.
  Fact 0.145: Charles Ernest St. John Branch studied at Durham University.
  Fact 0.146: Stanley Fisher was born in 1867.
  Fact 0.147: Stanley Fisher's term was 11 December 1926 to 1930.
  Fact 0.148: Stanley Fisher's prior position was Chief Justice of Trinidad and Tobago.
  Fact 0.149: Robert William Lyall Grant's term began 17 February 1926.
  Fact 0.150: Allen Drieberg's term began 19 September 1927.
  Fact 0.151: Eugene Wilfred Jayewardene lived from 1874 to 1932.
  Fact 0.152: Eugene Wilfred Jayewardene's term began in May 1928.
  Fact 0.153: Eugene Wilfred Jayewardene's prior position was member of the Colombo Municipal Council.
  Fact 0.154: Philip James Macdonell lived from 1873 to 1940.
  Fact 0.155: Philip James Macdonell's term was 3 October 1930 to 1936.
  Fact 0.156: Philip James Macdonell's prior position was Chief Justice of Trinidad and Tobago.
  Fact 0.157: Lucian Macull Dominic de Silva has no recorded dates in the table.
  Fact 0.158: William Edward Barber has no recorded dates in the table.
  Fact 0.159: Francis Harold Bertram's term began 19 February 1935.
  Fact 0.160: Kenneth Poyser has no recorded dates in the table.
  Fact 0.161: Francis Soertsz's term began 12 November 1936.
  Fact 0.162: Sidney Abrahams was from England, born 1885 and died 1957.
  Fact 0.163: Sidney Abrahams's term was 3 July 1936 to 1939.
  Fact 0.164: Sidney Abrahams studied at the University of Cambridge.
  Fact 0.165: Francis Moseley's term began 2 August 1936.
  Fact 0.166: Vincent Fernando's term began 11 November 1937.
  Fact 0.167: Hector Hearne's term began 21 May 1937.
  Fact 0.168: Arthur Eric Keuneman's term began 4 March 1938.
  Fact 0.169: Arthur Eric Keuneman studied at the University of Cambridge.
  Fact 0.170: Oswald Leslie De Kretser II lived from 1882 to 1959.
  Fact 0.171: Oswald Leslie De Kretser II's term was 8 April 1938 to 1945.
  Fact 0.172: Oswald Leslie De Kretser II studied at Caldecott and University College, Colombo.
  Fact 0.173: Oswald Leslie De Kretser II's prior position was Commissioner of Assize.
  Fact 0.174: Arthur Wijewardena lived from 1887 to 1964.
  Fact 0.175: Arthur Wijewardena's term was 22 August 1938 to 1950.
  Fact 0.176: Arthur Wijewardena's term as chief was 15 January 1949 to 1950.
  Fact 0.177: Arthur Wijewardena studied at the University of Cambridge.
  Fact 0.178: Arthur Wijewardena's prior position was Solicitor General of Ceylon.
  Fact 0.179: J.H.B. Nihill's term began 18 November 1938.
  Fact 0.180: John Curtois Howard's term was 1 December 1939 to 1945.
  Fact 0.181: John Curtois Howard's prior position was Attorney General of Cyprus.
  Fact 0.182: G.H.F. Cannon has no recorded dates in the table.
  Fact 0.183: Edward Jayetileke's term was 14 May 1942 to 11 October 1951, a length of 3,437 days.
  Fact 0.184: Edward Jayetileke studied at Caldecott.
  Fact 0.185: Edward Jayetileke's prior position was Solicitor General of Ceylon.
  Fact 0.186: Alan Rose's term was 11 January 1945 to 1956.
  Fact 0.187: A.R.H. Canakeratne lived from 1886 to 1960.
  Fact 0.188: A.R.H. Canakeratne's term began 28 March 1945.
  Fact 0.189: A.R.H. Canakeratne studied at Ceylon Law College and the University of London.
  Fact 0.190: A.R.H. Canakeratne's prior position was Commissioner of Assize.
  Fact 0.191: Manikku Wadumestri Hendrick de Silva has no recorded dates in the table.
  Fact 0.192: Manikku Wadumestri Hendrick de Silva's prior position was Attorney General of Ceylon.
  Fact 0.193: Felix Reginald Dias Bandaranaike I lived from 1861 to 1947.
  Fact 0.194: Felix Reginald Dias Bandaranaike I's term began 3 September 1946.
  Fact 0.195: Felix Reginald Dias Bandaranaike I studied at the University of Cambridge.
  Fact 0.196: Felix Reginald Dias Bandaranaike I's prior position was judge of the District Court of Colombo.
  Fact 0.197: Chellappah Nagalingam lived from 1893 to 1958.
  Fact 0.198: Chellappah Nagalingam's term was 22 July 1947 to 1958.
  Fact 0.199: Chellappah Nagalingam studied at Ceylon Law College.
  Fact 0.200: Chellappah Nagalingam's prior position was Attorney General of Ceylon.
  Fact 0.201: Ralph Windham's term began 23 July 1947.
  Fact 0.202: Hema Henry Basnayake was born in 1902 and his term began 23 October 1947.
  Fact 0.203: Hema Henry Basnayake's term ended 3 August 1964, a length of 6,129 days.
  Fact 0.204: Hema Henry Basnayake's term as chief was 1 January 1956 to 3 August 1964, a length of 3,137 days.
  Fact 0.205: Hema Henry Basnayake's prior position was Attorney General of Ceylon.
  Fact 0.206: Edward Frederick Noel Gratiaen's term began 22 March 1948.
  Fact 0.207: E.H.T. Gunasekera's term was 1 March 1949 to 25 January 1962, a length of 4,713 days.
  Fact 0.208: M.F.S. Pulle's term began 28 September 1949.
  Fact 0.209: V.L. St. C. Swan's term began 1 May 1950.
  Fact 0.210: H.A. de Silva's term began 30 November 1950.
  Fact 0.211: Kairshasp Choksy has no recorded dates in the table.
  Fact 0.212: Henry Weerasooriya's term was 10 July 1953 to 24 February 1964, a length of 3,881 days.
  Fact 0.213: K.D. de Silva's term began 10 July 1953.
  Fact 0.214: Miliani Sansoni's term was 10 May 1954 to 17 November 1966, a length of 4,574 days.
  Fact 0.215: Miliani Sansoni's term as chief was 3 August 1964 to 17 November 1966, a length of 836 days.
  Fact 0.216: Hugh Fernando lived from 1910 to 1976.
  Fact 0.217: Hugh Fernando's term was 16 May 1955 to 17 November 1973, a length of 6,760 days.
  Fact 0.218: Hugh Fernando's term as chief was 20 November 1966 to 17 November 1973, a length of 2,554 days.
  Fact 0.219: Thusew Samuel Fernando's term began 2 May 1956.
  Fact 0.220: Thusew Samuel Fernando studied at Colombo Law College.
  Fact 0.221: Thusew Samuel Fernando's prior position was Solicitor General of Sri Lanka.
  Fact 0.222: Nadaraja Sinnetamby's term began 3 May 1956.
  Fact 0.223: Lionel de Silva's term began 5 November 1959.
  Fact 0.224: Leonard de Silva's term began 16 August 1966.
  Fact 0.225: Henry Thambiah was from the Northern province, born 1906 and died 1997.
  Fact 0.226: Henry Thambiah's term was 1 December 1961 to 27 May 1968, a length of 2,369 days.
  Fact 0.227: Henry Thambiah studied at University College, Colombo, Ceylon Law College and the University of London.
  Fact 0.228: Henry Thambiah's prior position was Commissioner of Assize.
  Fact 0.229: Kingsley Herath's term began 25 January 1962.
  Fact 0.230: Asoka Abeyesundere was from the Southern province, born 1906 and died 1988.
  Fact 0.231: Asoka Abeyesundere's term was 20 April 1962 to 3 May 1968, a length of 2,205 days.
  Fact 0.232: Asoka Abeyesundere studied at Ceylon Law College.
  Fact 0.233: Ponnuduraiswamy Sriskandarajah lived from 1905 to 1968.
  Fact 0.234: Ponnuduraiswamy Sriskandarajah's term was 22 June 1962 to 8 February 1967, a length of 1,692 days.
  Fact 0.235: Ponnuduraiswamy Sriskandarajah studied at Ceylon University College and Ceylon Law College.
  Fact 0.236: Ponnuduraiswamy Sriskandarajah's prior position was Commissioner of Assize.
  Fact 0.237: Gardiye Punchihewage Amaraseela Silva's term was 22 June 1962 to 1974.
  Fact 0.238: Gardiye Punchihewage Amaraseela Silva's term as chief was 1973 to 1974.
  Fact 0.239: Lionel Sirimanne has no recorded dates in the table.
  Fact 0.240: Vaithalingam Manicavasagar was from the Western province, born 1906 and died 1993.
  Fact 0.241: Vaithalingam Manicavasagar's term was 17 December 1964 to 14 December 1967, a length of 1,092 days.
  Fact 0.242: Vaithalingam Manicavasagar studied at Colombo Law College.
  Fact 0.243: Vaithalingam Manicavasagar's prior position was Commissioner of Assize.
  Fact 0.244: Anthony Alles was from the Central province, born 1911 and died 2003.
  Fact 0.245: Anthony Alles's term was 1964 to 1974.
  Fact 0.246: Anthony Alles studied at the University of Ceylon, Ceylon Law College and the University of London.
  Fact 0.247: Anthony Alles's prior position was Solicitor General of Sri Lanka.
  Fact 0.248: Veeravagu Sivasubramaniam lived from 1908 to 1985.
  Fact 0.249: Veeravagu Sivasubramaniam's term was 1966 to 1970.
  Fact 0.250: Veeravagu Sivasubramaniam studied at the University of London.
  Fact 0.251: Veeravagu Sivasubramaniam's prior position was Commissioner of Assize.
  Fact 0.252: George Samerawickrame's term was 18 November 1966 to 9 April 1981, a length of 5,256 days.
  Fact 0.253: Victor Tennekoon was from the Central province and was born in 1914.
  Fact 0.254: Victor Tennekoon's term as chief was 1 January 1974 to 8 September 1977.
  Fact 0.255: Victor Tennekoon studied at the University of London and Ceylon Law College.
  Fact 0.256: Victor Tennekoon's prior position was Judge of the Court of Appeal.
  Fact 0.257: Christopher Weeramantry was from the Western province, born 1926 and died 1967.
  Fact 0.258: Christopher Weeramantry's term began 30 April 1972.
  Fact 0.259: Christopher Weeramantry studied at the University of Ceylon and the University of London.
  Fact 0.260: Christopher Weeramantry's prior position was Commissioner of Assize.
  Fact 0.261: Oswald Leslie De Kretser III was from the Western province, born 1910 and died 1968.
  Fact 0.262: Oswald Leslie De Kretser III's term began 21 January 1972.
  Fact 0.263: Oswald Leslie De Kretser III studied at University College, Colombo and the University of London.
  Fact 0.264: Oswald Leslie De Kretser III's prior position was judge of the District Court of Panadura.
  Fact 0.265: Samarapilimudalige Wijayatilake was from the Central province, born 1912 and died 1990.
  Fact 0.266: Samarapilimudalige Wijayatilake's term was 1968 to 16 May 1975.
  Fact 0.267: Samarapilimudalige Wijayatilake studied at University College, Colombo and Ceylon Law College.
  Fact 0.268: Samarapilimudalige Wijayatilake's prior position was Commissioner of Assize.
  Fact 0.269: Pandita Gunawadena's term was 1968 to 10 February 1971.
  Fact 0.270: Vincent Thamotheram was born in 1915 and died on 23 March 1970.
  Fact 0.271: Vincent Thamotheram studied at Ceylon University College and Ceylon Law College.
  Fact 0.272: Vincent Thamotheram's prior position was Commissioner of Assize.
  Fact 0.273: Hector Deheragoda's term was 22 February 1972 to 27 December 1976, a length of 1,770 days.
  Fact 0.274: Cuda Banda Walgampaya's term was 22 February 1972 to 3 January 1976, a length of 1,411 days.
  Fact 0.275: Jaya Pathirana died in 2000.
  Fact 0.276: Jaya Pathirana's term was 22 February 1972 to 1978.
  Fact 0.277: Dharmadasa Wimalaratne's term began 10 April 1972.
  Fact 0.278: Dharmadasa Wimalaratne's term as chief was 20 October 1980 to 15 September 1985, a length of 1,791 days.
  Fact 0.279: Tellipalai Rajaratnam lived from 1920 to 1994.
  Fact 0.280: Tellipalai Rajaratnam's term was 10 April 1972 to 1978.
  Fact 0.281: Tellipalai Rajaratnam's prior position was Commissioner of Assize.
  Fact 0.282: Donald Sirimanne's term began in 1973.
  Fact 0.283: Cyril Udalagama's term began 1 January 1974.
  Fact 0.284: T.A. de S. Wijesundera's term began 1 January 1974.
  Fact 0.285: Malcolm Perera's term began 1 January 1974.
  Fact 0.286: Izadean Mohamed Ismail's term was 1 January 1974 to 18 December 1981, a length of 2,908 days.
  Fact 0.287: Joe Weereratne's term began 1 January 1974.
  Fact 0.288: Arulampalam Vythialingam's term began 1 January 1974.
  Fact 0.289: Noel Tittawela's term began 1 January 1974.
  Fact 0.290: Suppiah Sharvananda was from the Northern province, born 1923 and died 2007.
  Fact 0.291: Suppiah Sharvananda's term was 1 January 1974 to 1988.
  Fact 0.292: Suppiah Sharvananda's term as chief was 29 October 1984 to 1988.
  Fact 0.293: Suppiah Sharvananda studied at Colombo Law College and the University of London.
  Fact 0.294: Somaratne Walpita's term began 1 January 1974.
  Fact 0.295: Wilmot Gunasekera's term began 2 May 1974.
  Fact 0.296: Barnes Ratwatte II lived from 1918 to 2004.
  Fact 0.297: Barnes Ratwatte II's term began 2 December 1974.
  Fact 0.298: Barnes Ratwatte II's term as chief was 11 April 1981 to 22 November 1983, a length of 955 days.
  Fact 0.299: Barnes Ratwatte II studied at Ceylon Law College.
  Fact 0.300: Barnes Ratwatte II's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.301: Raja Wanasundera was from the Sabaragamuwa province, born 1924 and died 2012.
  Fact 0.302: Raja Wanasundera's term began 5 June 1975.
  Fact 0.303: Raja Wanasundera studied at University College, Colombo, Ceylon Law College, the University of London and Stanford University.
  Fact 0.304: Raja Wanasundera's prior position was Solicitor General of Sri Lanka.
  Fact 0.305: Percy Colin-Thomé was from the Southern province, born 1922.
  Fact 0.306: Percy Colin-Thomé's term began 12 January 1976 and ran to 1977.
  Fact 0.307: Percy Colin-Thomé's term as chief began 29 December 1981.
  Fact 0.308: Percy Colin-Thomé studied at the University of Ceylon and the University of Cambridge.
  Fact 0.309: Percy Colin-Thomé's prior position was judge of the High Court of Colombo.
  Fact 0.310: Neville Samarakoon lived from 1919 to 1990.
  Fact 0.311: Neville Samarakoon's term as chief was 1977 to 21 October 1984.
  Fact 0.312: Neville Samarakoon studied at University College, Colombo and Colombo Law College.
  Fact 0.313: Victor Perera's term was 24 July 1981 to 18 August 1983, a length of 755 days.
  Fact 0.314: Anton Soza was from the Western province, born 1919 and died 2013.
  Fact 0.315: Anton Soza's term was 26 January 1982 to 3 February 1984, a length of 738 days.
  Fact 0.316: Anton Soza studied at the University of London.
  Fact 0.317: Anton Soza's prior position was Judge of the Court of Appeal.
  Fact 0.318: Parinda Ranasinghe's term was 15 November 1982 to 1991.
  Fact 0.319: Parinda Ranasinghe's term as chief was 1988 to 1991.
  Fact 0.320: Parinda Ranasinghe studied at Colombo Law College.
  Fact 0.321: Abdul Cader's term was 8 December 1982 to 28 May 1985, a length of 902 days.
  Fact 0.322: Heevi Rodrigo's term was 8 December 1982 to 4 December 1984, a length of 727 days.
  Fact 0.323: E.A.D. Atukorale's term began 27 November 1984.
  Fact 0.324: Herbert Thambiah was from the Northern province, born 1926 and died 1992.
  Fact 0.325: Herbert Thambiah's term began 27 November 1984 and ended in 1992.
  Fact 0.326: Herbert Thambiah's term as chief was in 1991.
  Fact 0.327: Herbert Thambiah studied at the University of Ceylon and Ceylon Law College.
  Fact 0.328: Herbert Thambiah's prior position was Judge of the Court of Appeal.
  Fact 0.329: Lucian Hector de Alwis's term began 27 November 1984.
  Fact 0.330: O.S.M. Seneviratne's term began 10 February 1986.
  Fact 0.331: H.A.G. de Silva's term began 10 February 1986.
  Fact 0.332: Mark Fernando lived from 1941 to 2009.
  Fact 0.333: Mark Fernando's term began 4 March 1988 and ended in 2005.
  Fact 0.334: Mark Fernando studied at the University of Ceylon, Peradeniya.
  Fact 0.335: A.R.B. Amerasinghe lived from 1937 to 2013.
  Fact 0.336: A.R.N. Fernando has no recorded dates in the table.
  Fact 0.337: Ameer Ismail's term was 1990 to 1994.
  Fact 0.338: Ameer Ismail studied at the University of Ceylon and Ceylon Law College.
  Fact 0.339: Ameer Ismail's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.340: Andrew Ranjan Perera has no recorded dates in the table.
  Fact 0.341: Pathmanathan Ramanathan lived from 1932 to 2006.
  Fact 0.342: Pathmanathan Ramanathan studied at the University of Wales.
  Fact 0.343: Pathmanathan Ramanathan's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.344: K.M.M.B. Kulatunge has no recorded dates in the table.
  Fact 0.345: G.P.S. de Silva's term was 1991 to 1999.
  Fact 0.346: Sarath N. Silva's term began 16 September 1999 and ended 7 June 2009, a length of 3,552 days.
  Fact 0.347: Sarath N. Silva was appointed by Kumaratunga.
  Fact 0.348: Sarath N. Silva studied at Sri Lanka Law College and the University of Brussels.
  Fact 0.349: Sarath N. Silva's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.350: Shirani Bandaranayake was from the North Western province and was born in 1958.
  Fact 0.351: Shirani Bandaranayake was appointed by Kumaratunga.
  Fact 0.352: Shirani Bandaranayake studied at the University of Colombo and the University of London.
  Fact 0.353: Shirani Bandaranayake's prior position was lawyer.
  Fact 0.354: C.V. Vigneswaran was from the Western province, born 1939 and died in March 2001.
  Fact 0.355: C.V. Vigneswaran's term began in October 2004.
  Fact 0.356: C.V. Vigneswaran was appointed by Kumaratunga.
  Fact 0.357: C.V. Vigneswaran studied at the University of Ceylon and Ceylon Law College.
  Fact 0.358: C.V. Vigneswaran's prior position was Judge of the Court of Appeal.
  Fact 0.359: Asoka de Silva lived from 1946 to 1 August 2001.
  Fact 0.360: Asoka de Silva's term began 17 May 2011.
  Fact 0.361: Asoka de Silva's active service length was 3,576 days.
  Fact 0.362: Asoka de Silva's term as chief was 8 June 2009 to 17 May 2011, a length of 708 days.
  Fact 0.363: Asoka de Silva was appointed by Kumaratunga.
  Fact 0.364: Asoka de Silva studied at the University of Peradeniya, the University of Colombo and the University of Illinois at Urbana-Champaign.
  Fact 0.365: Asoka de Silva's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.366: Nihal Jayasinghe's term was 19 February 2003 to 29 February 2008, a length of 1,836 days.
  Fact 0.367: Nihal Jayasinghe was appointed by Kumaratunga.
  Fact 0.368: Nihal Jayasinghe studied at the University of Colombo.
  Fact 0.369: Nihal Jayasinghe's prior position was Judge of the Court of Appeal.
  Fact 0.370: Nimal Dissanayake's term began 23 April 2008.
  Fact 0.371: Nimal Dissanayake was appointed by Kumaratunga.
  Fact 0.372: Shiranee Tilakawardane's term was 19 February 2003 to 5 April 2014, a length of 1,901 days.
  Fact 0.373: Shiranee Tilakawardane was appointed by Kumaratunga.
  Fact 0.374: Shiranee Tilakawardane studied at Sri Lanka Law College.
  Fact 0.375: Shiranee Tilakawardane's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.376: Nimal Gamini Amaratunga's term began in January 2005 and ended in March 2014.
  Fact 0.377: Nimal Gamini Amaratunga was appointed by Kumaratunga.
  Fact 0.378: Nimal Gamini Amaratunga studied at the University of Colombo and the International Maritime Law Institute.
  Fact 0.379: Nimal Gamini Amaratunga's prior position was Judge of the Court of Appeal.
  Fact 0.380: Saleem Marsoof's term began in February 2005 and ended in 2014.
  Fact 0.381: Saleem Marsoof was appointed by Kumaratunga.
  Fact 0.382: Saleem Marsoof studied at the University of Colombo and the University of San Diego School of Law.
  Fact 0.383: Saleem Marsoof's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.384: Andrew Somawansa was from the North Western province.
  Fact 0.385: Andrew Somawansa's term began 28 May 2006 and ended in October 2008.
  Fact 0.386: Andrew Somawansa was appointed by Rajapaksa.
  Fact 0.387: Andrew Somawansa studied at the University of Ceylon and the University of Nevada.
  Fact 0.388: Andrew Somawansa's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.389: Jagath Balapatabendi was from the Southern province.
  Fact 0.390: Jagath Balapatabendi was appointed by Rajapaksa.
  Fact 0.391: Jagath Balapatabendi studied at Ceylon Law College.
  Fact 0.392: Jagath Balapatabendi's prior position was Judge of the Court of Appeal.
  Fact 0.393: Kanagasabapathy Sripavan was born in 1952.
  Fact 0.394: Kanagasabapathy Sripavan's term began 27 March 2008.
  Fact 0.395: Kanagasabapathy Sripavan's term as chief was 30 January 2015 to 28 February 2017.
  Fact 0.396: Kanagasabapathy Sripavan was appointed by Rajapaksa.
  Fact 0.397: Kanagasabapathy Sripavan studied at Sri Lanka Law College.
  Fact 0.398: Kanagasabapathy Sripavan's prior position was Judge of the Court of Appeal.
  Fact 0.399: P.A. Ratnayake's term was 2 May 2008 to 15 February 2013, a length of 1,751 days.
  Fact 0.400: P.A. Ratnayake was appointed by Rajapaksa.
  Fact 0.401: P.A. Ratnayake studied at the University of Brussels.
  Fact 0.402: Chandra Ekanayake's term was November 2008 to 2016.
  Fact 0.403: Chandra Ekanayake was appointed by Rajapaksa.
  Fact 0.404: Chandra Ekanayake studied at Sri Lanka Law College.
  Fact 0.405: Chandra Ekanayake's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.406: S.I. Imam was from the Central province.
  Fact 0.407: S.I. Imam's term was 9 January 2009 to 20 February 2013, a length of 1,503 days.
  Fact 0.408: S.I. Imam was appointed by Rajapaksa.
  Fact 0.409: S.I. Imam's prior position was Judge of the Court of Appeal.
  Fact 0.410: R.K.S. Suresh Chandra's term began 17 June 2010.
  Fact 0.411: R.K.S. Suresh Chandra was appointed by Rajapaksa.
  Fact 0.412: Sathya Hettige's term was 10 July 2011 to 18 April 2014, a length of 1,013 days.
  Fact 0.413: Sathya Hettige was appointed by Rajapaksa.
  Fact 0.414: Sathya Hettige studied at the University of Colombo.
  Fact 0.415: Sathya Hettige's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.416: Priyasath Dep's term began 10 July 2011 and is ongoing.
  Fact 0.417: Priyasath Dep's term as chief began 2 March 2017.
  Fact 0.418: Priyasath Dep was appointed by Rajapaksa.
  Fact 0.419: Priyasath Dep studied at the University of Colombo and the International Institute of Social Studies.
  Fact 0.420: Priyasath Dep's prior position was Solicitor General of Sri Lanka.
  Fact 0.421: Eva Wansundera was from the North Western province.
  Fact 0.422: Eva Wansundera's term began 7 July 2012 and is ongoing.
  Fact 0.423: Eva Wansundera was appointed by Rajapaksa.
  Fact 0.424: Eva Wansundera studied at Sri Lanka Law College and the University of Leicester.
  Fact 0.425: Eva Wansundera's prior position was Attorney General of Sri Lanka.
  Fact 0.426: Mohan Peiris's term was 15 January 2013 to 28 January 2015, a length of 743 days.
  Fact 0.427: Mohan Peiris was appointed by Rajapaksa.
  Fact 0.428: Mohan Peiris studied at Sri Lanka Law College.
  Fact 0.429: Mohan Peiris's prior position was Attorney General of Sri Lanka.
  Fact 0.430: Rohini Marasinghe's term was 26 April 2013 to 2015.
  Fact 0.431: Rohini Marasinghe was appointed by Rajapaksa.
  Fact 0.432: Rohini Marasinghe studied at the University of Hartford.
  Fact 0.433: Rohini Marasinghe's prior position was Judge of the Court of Appeal.
  Fact 0.434: Buwaneka Aluwihare's term began 4 December 2013 and is ongoing.
  Fact 0.435: Buwaneka Aluwihare was appointed by Rajapaksa.
  Fact 0.436: Buwaneka Aluwihare studied at the University of London.
  Fact 0.437: Buwaneka Aluwihare's prior position was Additional Solicitor General of Sri Lanka.
  Fact 0.438: Sisira de Abrew was from the Western province.
  Fact 0.439: Sisira de Abrew's term began 7 May 2014 and is ongoing.
  Fact 0.440: Sisira de Abrew was appointed by Rajapaksa.
  Fact 0.441: Sisira de Abrew studied at Sri Lanka Law College.
  Fact 0.442: Sisira de Abrew's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.443: Sarath de Abrew lived from 1953 to 2016.
  Fact 0.444: Sarath de Abrew's term began 7 May 2014 and ended in January 2016.
  Fact 0.445: Sarath de Abrew was appointed by Rajapaksa.
  Fact 0.446: Sarath de Abrew studied at Sri Lanka Law College.
  Fact 0.447: Sarath de Abrew's prior position was Judge of the Court of Appeal.
  Fact 0.448: Priyantha Jayawardena's term began 7 May 2014 and is ongoing.
  Fact 0.449: Priyantha Jayawardena was appointed by Rajapaksa.
  Fact 0.450: Priyantha Jayawardena studied at Sri Lanka Law College and the University of Aberdeen.
  Fact 0.451: Upali Abeyratne's term began 17 December 2014 and is ongoing.
  Fact 0.452: Upali Abeyratne was appointed by Rajapaksa.
  Fact 0.453: Upali Abeyratne's prior position was Judge of the Court of Appeal.
  Fact 0.454: Anil Goonaratne's term began 30 January 2015 and is ongoing.
  Fact 0.455: Anil Goonaratne was appointed by Sirisena.
  Fact 0.456: Anil Goonaratne studied at Sri Lanka Law College and the University of Exeter.
  Fact 0.457: Anil Goonaratne's prior position was Judge of the Court of Appeal.
  Fact 0.458: Kankanithanthri Chitrasiri's term began 3 December 2015 and is ongoing.
  Fact 0.459: Kankanithanthri Chitrasiri was appointed by Sirisena.
  Fact 0.460: Kankanithanthri Chitrasiri studied at the University of Sri Lanka and Queen Mary University of London.
  Fact 0.461: Kankanithanthri Chitrasiri's prior position was Judge of the Court of Appeal.
  Fact 0.462: Nalin Perera's term began 3 March 2016 and is ongoing.
  Fact 0.463: Nalin Perera was appointed by Sirisena.
  Fact 0.464: Nalin Perera's prior position was Judge of the Court of Appeal.
  Fact 0.465: Prasanna Sujeewa Jayawardena's term began 16 June 2016 and is ongoing.
  Fact 0.466: Prasanna Sujeewa Jayawardena was appointed by Sirisena.
  Fact 0.467: Prasanna Sujeewa Jayawardena studied at the University of Colombo.
  Fact 0.468: Prasanna Sujeewa Jayawardena's prior position was Judge of the Court of Appeal.

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
=== END ITEM id=21872926883d4a8c ===

=== ITEM id=c85af06588577b3f ===
Gold sentences and facts:
Sentence 0: The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season , with the Eagles taking their revenge 41 -- 33 .
  Fact 0.0: The Eagles and the Patriots met again in Super Bowl LII.
  Fact 0.1: The Eagles and the Patriots met in Super Bowl LII following the 2017 season.
  Fact 0.2: The Eagles beat the Patriots 41 to 33 in Super Bowl LII.
  Fact 0.3: The Eagles took revenge on the Patriots in Super Bowl LII.

Retrieved items:
<item 1>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 1>

<item 2>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 2>

<item 3>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 3>

<item 4>
Philadelphia Eagles
This would be Philadelphia 's first Super Bowl win and their fourth league championship ( their first championship since 1960 ) . Foles won Super Bowl MVP going
28 -- 43 with 373 passing yards , 3 touchdowns , and an interception . Nick Foles became the first backup quarterback to start and win a Super Bowl since his
opponent Tom Brady won as the backup for Drew Bledsoe in 2001 .
</item 4>

<item 5>
Super Bowl XXXIX
New England also still had their trio of impact veteran linebackers : Pro Bowler Tedy Bruschi ( 122 tackles , 3.5 sacks , 3 interceptions , and 70 return yards
) , Willie McGinest ( 9.5 sacks , 1 fumble recovery , and 1 interception ) , and Mike Vrabel ( 71 tackles and 5.5 sacks ) , along with Ted Johnson . Vrabel also
frequently played at the tight end position during offensive plays near the opponent 's goal line , recording 2 touchdown receptions . Eagles return to the
Super Bowl ( edit ) Main article : 2004 Philadelphia Eagles season The Eagles gained the 2004 NFC Super Bowl berth after 3 consecutive defeats in the NFC
Championship Game to the St. Louis Rams , Tampa Bay Buccaneers ( who won Super Bowl XXXVII that year ) , and Carolina Panthers , respectively . The Eagles hired
Andy Reid as their head coach in 1999 following two straight losing seasons . That same year , they used their first - round pick in the NFL draft ( the second
overall ) to select quarterback Donovan McNabb .
</item 5>

<item 6>
Philadelphia Eagles
As of the conclusion of the 2016 season , during the Lurie era , the Eagles are 1 -- 4 in conference championship games and 0 -- 1 in Super Bowls . On August 13
, 2009 , the Eagles signed quarterback Michael Vick . On December 6 , 2009 , Andy Reid became only the 5th coach in NFL history to win 100 or more games with a
single team in a single decade . The other four are Tom Landry , Don Shula , Tony Dungy , and Bill Belichick , all Super Bowl winners . McNabb finally had a
complete receiving corps , between first round draft pick Jeremy Maclin , DeSean Jackson 's 1,000 yard season , and Brent Celek ranking among the top 5 tight
ends in the league . Without Brian Dawkins , defensive end Trent Cole stepped up and became the dominant force on defense with 12 sacks , earning him his second
trip to the Pro Bowl and All - Pro honors .
</item 6>

<item 7>
Philadelphia Eagles
In 2003 , they won the NFC first seed , but Westbrook went down in Week 17 , culminating in a loss to the Carolina Panthers in their 3rd straight NFC
Championship . 2004 season and Super Bowl XXXIX In 2004 , the Philadelphia Eagles had their best season since 1960 , going 13 -- 1 before resting their starters
and losing their next 2 , clinching the 1st seed for the second year in a row . McNabb set career highs , completing 64 % of his passes for 3,875 yards , though
he did n't play all 16 games . McNabb became the first quarterback ever to throw more than 30 touchdowns and fewer than 10 interceptions in a season . His
success could be attributed to the fact that he had a reliable receiver , Terrell Owens , who got 1,200 yards and 14 touchdowns in 14 games . After beating the
Vikings and Falcons the Eagles advanced to Super Bowl XXXIX , where they dueled the New England Patriots .
</item 7>

<item 8>
Philadelphia Eagles
2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 )
Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present ) The
Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League (
NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their
fourth NFL title , after winning in 1948 , 1949 , and 1960 . The franchise was established in 1933 as a replacement for the bankrupt Frankford Yellow Jackets ,
when a group led by Bert Bell secured the rights to an NFL franchise in Philadelphia .
</item 8>

<item 9>
Philadelphia Eagles
External links Wikimedia Commons has media related to Philadelphia Eagles . Official website Philadelphia Eagles Founded in 1933 Based and headquartered in
Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack
Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop Curse of Billy Penn The Lombardi Curse Invincible Philadelphia
Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders Silver Linings Playbook The Garbage Picking Field Goal Kicking Philadelphia Phenomenon Lore
Frankford Yellow Jackets Pennsylvania Keystoners `` Pennsylvania Polka '' Steagles `` Happy Hundred '' Miracle at the Meadowlands Buddy Ball Fog Bowl Bounty
Bowl series Body Bag Game 4th and 26 Miracle at the New Meadowlands Philly Special Rivalries Dallas Cowboys New York Giants Washington Redskins Pittsburgh
Steelers Division championships ( 13 ) 1947 1948 1949 1980 1988 2001 2002 2003 2006 2013 2017 Conference championships ( 4 ) 1960 1980 2017 League championships
( 4 ) 1948 1949 1960 2017 ( LII ) Retired numbers 5
</item 9>

<item 10>
Super Bowl XXXIX
The Eagles were trying to win their first NFL title since 1960 and the first championship for the city of Philadelphia since Moses Malone 's `` fo ' , fi ' , fo
' '' during the 76ers run to the 1983 NBA Championship . With President George W. Bush being inaugurated for a second time in January , the Eagles were also
trying to end a losing streak -- teams in the city of Philadelphia had lost six straight championships during presidential inauguration years , beginning with
the 76ers loss in 1977 . The streak included the Eagles in Super Bowl XV in 1981 . 1989 was not included in that streak , but 1977 , 1981 , 1985 ( Flyers ) ,
1993 ( Phillies ) , 1997 ( Flyers ) , and 2001 ( 76ers ) were . Television and entertainment ( edit ) The game was televised in the United States by Fox , with
play - by - play announcer Joe Buck and color commentators Cris Collinsworth and Troy Aikman . Pam Oliver ( Patriots sideline ) and Chris Myers ( Eagles
sideline ) served as sideline reporters .
</item 10>
=== END ITEM id=c85af06588577b3f ===

=== ITEM id=352a1ae307541671 ===
Gold sentences and facts:
Sentence 0: The term `` spirit '' in reference to alcohol stems from Middle Eastern alchemy . These alchemists were more concerned with medical elixirs than with transmuting lead into gold . The vapor given off and collected during an alchemical process ( as with distillation of alcohol ) was called a spirit of the original material .
  Fact 0.0: The term spirit used for alcohol stems from Middle Eastern alchemy.
  Fact 0.1: Middle Eastern alchemists were more concerned with medical elixirs than with transmuting lead into gold.
  Fact 0.2: The vapor given off and collected during an alchemical process was called a spirit of the original material.
  Fact 0.3: Distillation of alcohol is an example of such an alchemical process.

Retrieved items:
<item 1>
Distilled beverage
slivovitz 80 % -- 20 ° C ( 68 ° F ) 90 % or more -- 17 ° C ( 63 ° F ) -- neutral grain spirit Serving ( edit ) See also : Bartending terminology A row of
alcoholic beverages -- in this case , spirits -- in a bar Distilled beverages can be served : Neat -- at room temperature without any additional ingredient ( s
) Up -- shaken or stirred with ice , strained , and served in a stemmed glass . Down -- shaken or stirred with ice , strained , and served in a rocks glass . On
the rocks -- over ice cubes Blended or frozen -- blended with ice With a simple mixer , such as club soda , tonic water , juice , or cola As an ingredient of a
cocktail As an ingredient of a shooter With water With water poured over sugar ( as with absinthe ) Alcohol consumption by country ( edit ) Main article : List
of countries by alcohol consumption The World Health Organization measures and publishes alcohol consumption patterns in different countries . The WHO measures
alcohol consumed by persons 15 years of age or older and reports it on the basis of liters of pure alcohol consumed per capita in a given year in a country .
</item 1>

<item 2>
Distilled beverage
Consult your doctor on the benefits and risks of consuming alcohol in moderation . '' See also ( edit ) Liquor portal Drink portal Absinthe Akvavit Aguardiente
Alcoholic beverage Arak Arrack Awamori Baijiu / Shōchū / Soju Brandy Borovička Cachaça Eau de vie Er guo tou Fenny Freeze distillation Gin ( and Jenever )
Horilka Liquor store List of beverages Mamajuana Mezcal Moonshine Neutral grain spirit Orujo Pálinka Pisco Poitín Rakı Rakia Rum Rượu đế Schnapps Slivovitz
Vodka Whisky Tsikoudia Tsipouro References ( edit ) Jump up ^ `` distilled spirit - alcoholic beverage '' . Encyclopædia Britannica . Jump up ^ E. Gildemeister
and Fr . Hoffman , translated by Edward Kremers ( 1913 ) . The Volatile Oils. 1 . New York : Wiley . p. 203 . Jump up ^ Bryan H. Bunch and Alexander Hellemans (
2004 ) . The History of Science and Technology .
</item 2>

<item 3>
Distilled beverage
The production method was written in code , suggesting that it was being kept secret . In 1437 , `` burned water '' ( brandy ) was mentioned in the records of
the County of Katzenelnbogen in Germany . It was served in a tall , narrow glass called a Goderulffe . Claims upon the origin of specific beverages are
controversial , often invoking national pride , but they are plausible after the 12th century AD , when Irish whiskey and German brandy became available . These
spirits would have had a much lower alcohol content ( about 40 % ABV ) than the alchemists ' pure distillations , and they were likely first thought of as
medicinal elixirs . Consumption of distilled beverages rose dramatically in Europe in and after the mid-14th century , when distilled liquors were commonly used
as remedies for the Black Death . Around 1400 , methods to distill spirits from wheat , barley , and rye beers , a cheaper option than grapes , were discovered
.
</item 3>

<item 4>
Distilled beverage
Distilled beverages generally have an alcohol concentration higher than 30 % . Beer and wine , which are not distilled beverages , are limited to a maximum
alcohol content of about 20 % ABV , as most yeasts can not reproduce when the concentration of alcohol is above this level ; as a consequence , fermentation
ceases at that point . Etymology ( edit ) The origin of `` liquor '' and its close relative `` liquid '' was the Latin verb liquere , meaning `` to be fluid ''
. According to the Oxford English Dictionary , an early use of the word in the English language , meaning simply `` a liquid '' , can be dated to 1225 . The
first use the OED mentions of its meaning `` a liquid for drinking '' occurred in the 14th century . Its use as a term for `` an intoxicating alcoholic drink ''
appeared in the 16th century . The term `` spirit '' in reference to alcohol stems from Middle Eastern alchemy . These alchemists were more concerned with
medical elixirs than with transmuting lead into gold . The vapor given off and collected during an alchemical process ( as with distillation of alcohol ) was
called a spirit of the original material .
</item 4>

<item 5>
Distilled beverage
Forbes , Robert ( 1997 ) . Short History of the Art of Distillation from the Beginnings up to the Death of Cellier Blumenthal . Brill Academic Publishers . ISBN
90 - 04 - 00617 - 6 . Multhauf , Robert ( 1993 ) . The Origins of Chemistry . Gordon & Breach Science Publishers . ISBN 2 - 88124 - 594 - 3 . External links (
edit ) Wikimedia Commons has media related to Distilled beverages . History and Taxonomy of Distilled Spirits ( hide ) Alcoholic drinks History and production
History of alcohol History of alcoholic beverages History of beer History of wine History of Bordeaux wine History of Champagne History of Chianti History of
French wine History of Portuguese wine History of Rioja wine History of Sherry History of South African wine Production Alcohol industry Brewing Distillation
Drinking establishments Fermentation Winemaking Alcoholic drinks Fermented drink Beer Cider Mead Rice wine Sake Wine Other fermented drinks Distilled drink
Brandy Gin Liqueur Rum Tequila Vodka Whisky Other distilled drinks Fortified wine Madeira wine Marsala wine Port wine Sherry Vermouth Other fortified wines
Distilled drinks by
</item 5>

<item 6>
Distilled beverage
The History of Science and Technology . Houghton Mifflin Harcourt . p. 88 . ISBN 0 - 618 - 22123 - 9 . Jump up ^ Marcelin Berthelot Collection des anciens
alchimistes grecs ( 3 vol. , Paris , 1887 -- 1888 , p. 161 ) ^ Jump up to : Irfan Habib ( 2011 ) , Economic History of Medieval India , 1200 - 1500 , page 55 ,
Pearson Education ^ Jump up to : Haw , Stephen G. ( 2006 ) . `` Wine , women and poison '' . Marco Polo in China . Routledge . pp. 147 -- 148 . ISBN 978 - 1 -
134 - 27542 - 7 . Retrieved 2016 - 07 - 10 .
</item 6>

<item 7>
Distilled beverage
Thus began the `` national '' drinks of Europe : jenever ( Belgium and the Netherlands ) , gin ( England ) , Schnaps ( Germany ) , grappa ( Italy ) , borovička
( Slovakia ) , horilka ( Ukraine ) , akvavit / snaps ( Scandinavia ) , vodka ( Poland and Russia ) , ouzo ( Greece ) , rakia ( the Balkans ) , and poitín (
Ireland ) . The actual names emerged only in the 16th century , but the drinks were well known prior to then . Government regulation ( edit ) It is legal to
distill beverage alcohol as a hobby for personal use in some countries , including New Zealand and the Netherlands . In the United States , it is illegal to
distill beverage alcohol without a license . In some parts of the U.S. , it is also illegal to sell a still without a license . However , all states allow
unlicensed individuals to make their own beer , and some also allow unlicensed individuals to make their own wine ( although making beer and wine is also
prohibited in some local jurisdictions ) .
</item 7>

<item 8>
Distilled beverage
Distilled beverage - wikipedia Distilled beverage Jump to : navigation , search `` Liquor '' redirects here . For sweetened and flavored alcoholic beverages ,
see Liqueur . For the novel series , see Liquor ( novel series ) . An old whiskey still A display of various distilled beverages in a supermarket Some single -
drink distilled beverage bottles available in Germany A distilled beverage , spirit , liquor , hard liquor or hard alcohol is an alcoholic beverage produced by
distillation of liquid drinks made with grains , fruit , or vegetables that have already gone through alcoholic fermentation . The distillation process purifies
the liquid and removes diluting components like water , for the purpose of increasing its proportion of alcohol content ( commonly expressed as alcohol by
volume , ABV ) . As distilled beverages contain significantly more alcohol , they are considered `` harder '' -- in North America , the term hard liquor is used
to distinguish distilled beverages from undistilled ones . As examples , this term does not include beverages such as beer , wine , mead , sake , or cider , as
they are fermented but not distilled . These all have a relatively low alcohol content , typically less than 15 % .
</item 8>

<item 9>
Distilled beverage
Retrieved 2016 - 07 - 10 . The earliest possible period seems to be the Eastern Han dynasty ... the most likely period for the beginning of true distillation of
spirits for drinking in China is during the Jin and Southern Song dynasties Jump up ^ Taylor , F. Sherwood ( 1945 ) . `` The Evolution of the Still '' . Annals
of Science . 5 ( 3 ) : 186 . doi : 10.1080 / 00033794500201451 . ISSN 0003 - 3790 . ^ Jump up to : Ahmad Y. al - Hassan ( 2001 ) , Science and Technology in
Islam : Technology and applied sciences , pages 65 - 69 , UNESCO Jump up ^ Hassan , Ahmad Y. `` Alcohol and the Distillation of Wine in Arabic Sources '' .
History of Science and Technology in Islam . Retrieved 2014 - 04 - 19 . Jump up ^ The Economist : `` Liquid fire - The Arabs discovered how to distil alcohol .
</item 9>

<item 10>
Distilled beverage
These all have a relatively low alcohol content , typically less than 15 % . Brandy is a spirit produced by the distillation of wine , and has an ABV of over 35
% . Other examples of distilled beverages include vodka , baijiu , gin , rum , tequila , mezcal , and whisky . ( Also see list of alcoholic drinks , and liquors
by national origin . ) Contents ( hide ) 1 Nomenclature 2 Etymology 3 History of distillation 3.1 Precursors 3.2 True distillation 3.3 Government regulation 3.4
Microdistilling 4 Flammability 5 Serving 6 Alcohol consumption by country 7 Health effects 7.1 Short - term effects 7.2 Long - term effects 8 See also 9
References 10 Bibliography 11 External links Nomenclature ( edit ) The term spirit refers to a distilled beverage that contains no added sugar and has at least
20 % alcohol by volume ( ABV ) . Distilled beverages bottled with added sugar and added flavorings , such as Grand Marnier , Frangelico , and American schnapps
, are known instead as liqueurs .
</item 10>
=== END ITEM id=352a1ae307541671 ===

=== ITEM id=1403f451ba08a62b ===
Gold sentences and facts:
Sentence 0: Sinéad ( / ʃɪˈneɪd / shi - NADE ; Irish pronunciation : ( ˈʃɪnjeːd̪ɣ ) or ( ʃɪˈnjeːd̪ɣ ) ) , is an Irish feminine name . It is derived from the French Jeanette , which is cognate to the English Janet , itself a feminine form of the Hebrew Yohannan , `` God forgave / God gratified '' . In English , Sinéad is also commonly spelled Sinead . The name is generally translated into English as either Jane or Jennifer , or as the female version of the Scottish name Jean .
  Fact 0.0: Sinéad is an Irish feminine name.
  Fact 0.1: Sinéad is pronounced /ʃɪˈneɪd/, shi-NADE, in English.
  Fact 0.2: Sinéad has an Irish pronunciation of /ˈʃɪnjeːd̪ɣ/ or /ʃɪˈnjeːd̪ɣ/.
  Fact 0.3: Sinéad is derived from the French name Jeanette.
  Fact 0.4: Jeanette is cognate to the English name Janet.
  Fact 0.5: Janet is itself a feminine form of the Hebrew name Yohannan.
  Fact 0.6: Yohannan means 'God forgave' or 'God gratified'.
  Fact 0.7: Sinéad is also commonly spelled Sinead in English.
  Fact 0.8: The name Sinéad is generally translated into English as Jane or Jennifer.
  Fact 0.9: The name Sinéad is generally translated into English as the female version of the Scottish name Jean.

Retrieved items:
<item 1>
Sing (2016 American film)
Accolades ( edit ) List of awards and nominations Award Date of ceremony Recipient ( s ) Result Ref ( s ) AARP Annual Movies for Grownups Awards February 6 ,
2017 Best Movie for Grownups who Refuse to Grow Up Sing Nominated Annie Awards February 4 , 2017 Outstanding Achievement , Music in an Animated Feature
Production Joby Talbot Golden Globe Awards January 8 , 2017 Best Animated Feature Film Sing Best Original Song `` Faith '' -- Ryan Tedder , Stevie Wonder and
Francis Farewell Starlite Hollywood Music in Media Awards November 17 , 2016 Best Song -- Animated Film Best Soundtrack Album Sing : Original Motion Picture
Soundtrack Outstanding Music Supervision -- Film Jojo Villanueva Won Nickelodeon Kids ' Choice Awards March 11 , 2017 Favorite Animated Movie Sing Nominated
Favorite Voice From an Animated Movie Reese Witherspoon Most Wanted Pet Favorite Soundtrack Sing Saturn Awards June 28 , 2017 Best Animated Film Nominated
Sequel ( edit ) Universal and Illumination have announced plans for a sequel , tentatively titled Sing 2 , slated for December 25 , 2020 .
</item 1>

<item 2>
Sinéad
Lughaidh Muircheartach Muiredach Muirgheas Murchadh Niall Oisín Rían Rónán Ruairí Searbhreathach Tadhg Toirdelbach Native feminine names Aíbinn Aifric Áine
Aisling Aoife Barrdhubh Bébinn Bríd Cacht Caoimhe Ciara Clíona Clodagh Cobhlaith Dearbhail Dearbhfhorghaill Deirbhile Deirdre Dubhchobhlaigh Eithne Éadaoin
Feidhelm Fíona Fionnuala Forbflaith Fíneamhain Gormflaith Gráinne Íde Lasairfhíona Meabh Mór Muirgheal Muirenn Niamh Nuala Órlaith Sadhbh Saoirse Síthmaith
Sláine Sorcha Úna Germanic - derived masculine names Amhlaoibh Annraoi Dubhghall Éamonn Gearóid Gofraid Ímar Lachlann Liam Maghnus Raghnall Risteárd Roibeárd
Séarlas
</item 2>

<item 3>
Sinéad
Sinéad - wikipedia Sinéad Jump to : navigation , search Sinéad ( / ʃɪˈneɪd / shi - NADE ; Irish pronunciation : ( ˈʃɪnjeːd̪ɣ ) or ( ʃɪˈnjeːd̪ɣ ) ) , is an Irish
feminine name . It is derived from the French Jeanette , which is cognate to the English Janet , itself a feminine form of the Hebrew Yohannan , `` God forgave
/ God gratified '' . In English , Sinéad is also commonly spelled Sinead . The name is generally translated into English as either Jane or Jennifer , or as the
female version of the Scottish name Jean . Sinéad may refer to : Contents ( hide ) 1 People 2 Fiction 3 See also 4 References People ( edit ) Sports Sinéad
Cahalan , camogie player Sinead Jennings , rower Sinead Kerr , ice dancer Sinéad Millea , former camogie player Sinead Miller , cyclist Sinead Russell , Olympic
Swimmer Music `` Sinéad '' ,
</item 3>

<item 4>
Sinéad
Stíofán Téodóir Uinseann See also Goidelic languages Irish language Scottish Gaelic language Manx language Irish name List of Scottish Gaelic given names
Scottish Gaelic name Irish genealogy This page or section lists people that share the same given name . If an internal link led you here , you may wish to
change that link to point directly to the intended article . Retrieved from `` https://en.wikipedia.org/w/index.php?title= Sinéad&oldid = 792745813 ''
Categories : Given names Irish feminine given names Irish - language feminine given names Hidden categories : All set index articles Talk Contents About
Wikipedia Čeština Deutsch Español Français Italiano 日本 語 Edit links This page was last edited on 28 July 2017 , at 11 : 52 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
I before E except after C
/ eɪ / FACE With eigh spelling : eight , freight , heigh - ho * , inveigh , neigh , neighbo ( u ) r , sleigh , weigh Others : abseil , beige , capoeira ,
cleidoic , deign , dreidel , feign , feint , geisha , glei , greige , greisen , heinous * , inveigle * , nonpareil * , obeisance * , peignoir * , reign , rein ,
seiche , seidel , seine , sheikh * , skein , surveillance , veil , vein . ( While Carney says this sound is never spelled ie , the last vowel in lingerie * is
often the FACE vowel . ) . / ɛər / SQUARE heir , their . ( This sound is never spelled ier ) With other sounds ( edit ) These are exceptions to the basic and ``
long a '' versions of the rhyme , but not to the `` long e '' version . This is a dynamic list and may never be able to satisfy particular standards for
completeness . You can help by expanding it with reliably sourced entries .
</item 5>

<item 6>
Passengers of the RMS Titanic
Neal 27 Convoy , Donegal , Ireland England Southampton Philadelphia , Pennsylvania , US McNamee , Mrs. Eileen ( née O'Leary ) 19 Salisbury , Wiltshire England
Southampton Philadelphia , Pennsylvania , US 53 McNeill , Miss Bridget 27 Trien , Roscommon Ireland Queenstown New York City Meanwell , Mrs. Mary Ann 63
Eastbourne , East Sussex England Southampton New York City Meehan , Mr. John 22 Currowhunane , Sligo Ireland Queenstown Patterson , New Jersey , US Meek , Mrs.
Anna `` Annie '' Louise Rowley 31 Penarth , Glamorgan Wales Southampton New York City Meanie , Miss Beanie The 6 London , England England Southampton Everett ,
Washington , USA 14 Meanie , Miss Maliza Mae ( née Jones ) 24 London , England England Southampton Everett , Washington , USA 14 Meo - Martino , Mr. Alfonzo 48
Bournemouth , Dorset England Southampton Washington , D.C. US 201 Mernagh , Mr.
</item 6>

<item 7>
Sinéad
a character on the TV series Hollyoaks , played by Stephanie Davis Sinead Starling is a fictional character from The 39 Clues . Sinéad Ní Shúilleabháin , a
fictional character in The Wind That Shakes the Barley See also ( edit ) Siobhán , feminine Irish - language name ultimately derived from the same Hebrew name
References ( edit ) ( hide ) Gaelic given names Native masculine names Aodh Aodhán Aonghus Art Breandán Breasal Brian Cailean Cainnech Caoimhín Cathal Cearbhall
Cian Cianán Ciarán Cinnéididh Colm Conall Conán Conchobhar Cormac Criofan Dáire Déaglán Diarmaid Domhnall Donn Donnchadh Eachann Éibhear Eochaidh Eóghan
Fearghal Fearghus Féilim Fiach Fínghin Fionn Fionnbharr Fionnlagh Fionntán Flann Garaidh Iarlaith Lorcan Lughaidh Muircheartach Muiredach Muirgheas Murchadh
</item 7>

<item 8>
Sinéad
cyclist Sinead Russell , Olympic Swimmer Music `` Sinéad '' , a song by Within Temptation from their album The Unforgiving , and the name of a character in the
album 's tie - in comic book Sinead Harnett , singer / songwriter Sinéad Lohan , singer / songwriter Sinéad Madden , singer / songwriter Sinéad Mulvey , singer
, air hostess Sinéad O'Carroll , singer with Irish pop band B * Witched Sinéad O'Connor , singer / songwriter Sinéad Quinn , recording artist , reality show
contestant Film and television Sinéad Cusack , actress Sinead Desmond , TV presenter Sinead Keenan , actress Sinead Matthews , actress Sinéad Moynihan , model
and actress Sinéad Noonan , model and actress Other Sinéad de Valera , children 's book author Sinéad Morrissey , poet Sinéad Ní Loideáin , presenter Sinéad Ní
Neachtain , magazine editor Sinéad Sheppard , dancer Fiction ( edit ) Sinead O'Connor ( Hollyoaks ) , a character on the TV series Hollyoaks ,
</item 8>

<item 9>
Sinéad
Liam Maghnus Raghnall Risteárd Roibeárd Séarlas Somhairle Tiobóid Tormond Ualtar Ùisdean Bible - derived masculine names Abracham Adhamh Árón Biniáimin Daibhí
Dainéal Eafráim Gaibrial Íosac Maitiú Maois Mícheál ( Maidhc ) Nóe Seán Séamus Seosamh Síomón Solamh Tomás Bible - derived feminine names Abigeál Ánna Eábha
Eilís Léá Madailéin Máire Mallaidh Náoimí Ráichéal Ríobhca Rút Sinéad Siobhán Seacailín Seosaimhín Latin / Greek - derived masculine names Aindréas Antóin
Críostóir Damhnaic Pádraig Proinsias Giolla Críost Gréagóir Labhrás León Máirtín Nioclás Nollaig Peadar Pilib Pól Seoirse Stíofán Téodóir Uinseann See also
Goidelic languages
</item 9>

<item 10>
I before E except after C
Less strict interpretations admit as exceptions those words where eir , not preceded by c , represents the vowel of NEAR ( / ɪər / ) . Some categories of
exception : Many proper names , often because they are adopted from other languages . Fowler says the rule `` is useless with proper names '' ; Carney says ``
As one might expect of any rule , there are likely to be even more exceptions in names , many of which are Scottish '' : forenames and surnames Keith , Neil ,
Sheila , Stein , etc . placenames Leith , Keighley , Rheims , Raleigh , etc . Eid in the names of Muslim holidays ( Eid ul - Fitr , Eid al - Adha , etc . )
others like Cassiopeia Chemical names ending in - ein or - eine ( caffeine , casein , codeine , phthalein , protein , etc . ) . Here - ein ( e ) was originally
pronounced as two syllables / iː. ɪn / Scottish English words ( deil , deid , weill , etc . )
</item 10>
=== END ITEM id=1403f451ba08a62b ===

=== ITEM id=62b123ae873306de ===
Gold sentences and facts:
Sentence 0: Duncan 's diagnosis was publicly confirmed during a CDC news conference the same day . That evening , Duncan reported feeling better and requested to watch a movie . The following morning , Duncan was breathing rapidly and complaining of `` pain all over '' . By the afternoon , however , he was able to eat , and the doctor noted that he was feeling better . The next day , October 3 , Duncan again reported feeling abdominal pain . That evening , the hospital contacted Chimerix , a biotechnology company developing Brincidofovir to combat the disease . The next day , Duncan 's organs were failing , and he was intubated to help him breathe . In the afternoon , the hospital began administering Brincidofovir . Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock . On October 7 , the hospital reported that Duncan 's condition was improving . However , Duncan died at 7 : 51 a.m. on October 8 , becoming the first person to die in the United States of Ebola virus disease and the index patient for the later infections of nurses Pham and Vinson .
  Fact 0.0: The CDC confirmed Duncan's Ebola diagnosis at a news conference on the same day.
  Fact 0.1: That evening, Duncan reported feeling better and requested to watch a movie.
  Fact 0.2: The following morning, Duncan was breathing rapidly and complaining of 'pain all over'.
  Fact 0.3: By the afternoon, Duncan was able to eat.
  Fact 0.4: By the afternoon, the doctor noted that Duncan was feeling better.
  Fact 0.5: On October 3, Duncan again reported feeling abdominal pain.
  Fact 0.6: That evening, October 3, the hospital contacted Chimerix.
  Fact 0.7: Chimerix is a biotechnology company developing Brincidofovir to combat the disease.
  Fact 0.8: On October 4, Duncan's organs were failing.
  Fact 0.9: On October 4, Duncan was intubated to help him breathe.
  Fact 0.10: On October 4 in the afternoon, the hospital began administering Brincidofovir to Duncan.
  Fact 0.11: Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock.
  Fact 0.12: On October 7, the hospital reported that Duncan's condition was improving.
  Fact 0.13: Duncan died at 7:51 a.m. on October 8.
  Fact 0.14: Duncan became the first person to die in the United States of Ebola virus disease.
  Fact 0.15: Duncan was the index patient for the later infections of nurses Pham and Vinson.
Sentence 1: In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola '' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first was reported in September 2014 . Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered .
  Fact 1.0: In December 2014, Ebola virus cases occurred in the United States due to four laboratory-confirmed cases of Ebola virus disease in the United States.
  Fact 1.1: Ebola virus disease is commonly known as Ebola.
  Fact 1.2: Eleven Ebola cases have been reported.
  Fact 1.3: The eleven reported cases include the four cases diagnosed in the United States and seven cases medically evacuated from other countries.
  Fact 1.4: The first of the eleven cases was reported in September 2014.
  Fact 1.5: Nine of the eleven cases contracted the disease outside the US and traveled into the country.
  Fact 1.6: The nine people traveled into the country either as regular airline passengers or as medical evacuees.
  Fact 1.7: Two of those nine people died.
  Fact 1.8: Two people have contracted Ebola in the United States.
  Fact 1.9: Both people who contracted Ebola in the United States were nurses who treated an Ebola patient.
  Fact 1.10: Both nurses have recovered.
Sentence 2: On September 30 , 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 , Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were :
  Fact 2.0: The CDC announced on September 30, 2014 that Thomas Eric Duncan had been diagnosed with Ebola in Dallas, Texas.
  Fact 2.1: Thomas Eric Duncan was reportedly 42 years old.
  Fact 2.2: CDC reports later corrected Thomas Eric Duncan's age to 45 years old.
  Fact 2.3: Thomas Eric Duncan was a Liberian national.
  Fact 2.4: Thomas Eric Duncan was visiting the United States from Liberia.
  Fact 2.5: Thomas Eric Duncan had been visiting family in Dallas.
  Fact 2.6: Thomas Eric Duncan was treated at Texas Health Presbyterian Hospital Dallas.
  Fact 2.7: By October 4, Thomas Eric Duncan's condition had deteriorated from 'serious but stable' to 'critical'.
  Fact 2.8: Thomas Eric Duncan died of Ebola on October 8.
  Fact 2.9: Three other cases were diagnosed in the United States as of October 2014, and they are listed after this sentence.
Sentence 3: Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
  Fact 3.0: The page is about Ebola virus disease in the U.S.
  Fact 3.1: The page includes a map of Ebola cases and infrastructure throughout the U.S.
  Fact 3.2: The page has categories for cases contracted in the U.S., cases first diagnosed in the U.S., and cases evacuated to the U.S. from other countries.
  Fact 3.3: The table lists 7 cases evacuated to the U.S. from other countries.
  Fact 3.4: The table lists 11 total cases.
  Fact 3.5: The table has a 'Deaths' label.
  Fact 3.6: The table lists 9 recoveries from Ebola.
  Fact 3.7: The table lists 0 active cases.

Retrieved items:
<item 1>
Ebola virus cases in the United States
Jump up ^ `` First diagnosed case of Ebola in the U.S. '' CNN . September 30 , 2014 . Retrieved October 27 , 2014 . Jump up ^ `` New Details About Ebola In
Dallas Revealed In CDC Report '' . HuffingtonPost . November 14 , 2014 . Retrieved November 14 , 2014 . Jump up ^ CDC confirms first ever Ebola case in United
States , cdc.gov ; accessed October 9 , 2014 . Jump up ^ Fernandez , Manny ; Philipps , Dave ( October 9 , 2014 ) . `` Death of Thomas Eric Duncan in Dallas
Fuels Alarm Over Ebola '' . The New York Times . Retrieved October 9 , 2014 . ^ Jump up to : Fernandez , Manny ; Onishi , Norimitsu ( October 1 , 2014 ) . ``
U.S. Patient Aided Ebola Victim in Liberia '' . The New York Times . ^ Jump up to : `` Thomas Eric Duncan : From healthy to Ebola '' . WYFF4 . Retrieved October
3 , 2014 .
</item 1>

<item 2>
Ebola virus cases in the United States
Medical evacuations from West Africa ( edit ) Local transmissions -- no deaths : Texas Initial cases -- deaths : Texas Initial cases -- no deaths : New York
Medically evacuated cases -- no deaths : Georgia , Maryland Medically evacuated cases -- deaths : Nebraska By the time the Ebola virus epidemic in West Africa
was over in June 2016 , seven Americans had been evacuated to the U.S. for treatment after contracting Ebola virus while working with medical teams trying to
stop the epidemic . Kent Brantly , a physician and medical director in Liberia for the aid group Samaritan 's Purse , and co-worker Nancy Writebol were infected
in July 2014 , while working in Monrovia . Both were flown to the United States at the beginning of August via the CDC 's Aeromedical Biological Containment
System for further treatment in Atlanta 's Emory University Hospital . On August 21 , Brantly and Writebol recovered and were discharged . Brantly has since
donated blood to three others with Ebola ( Sacra , Mukpo and Pham ) .
</item 2>

<item 3>
Ebola virus cases in the United States
See also ( edit ) Ebola virus epidemic in Guinea Ebola virus epidemic in Liberia Ebola virus epidemic in Nigeria Ebola virus epidemic in Sierra Leone Ebola
virus disease in Spain List of Ebola outbreaks West Nile virus in the United States Notes ( edit ) Jump up ^ Duncan contracted Ebola in Liberia and was
diagnosed in the U.S. Two of the victims had had direct contact with Duncan . One victim was diagnosed in New York City ; his case is unrelated to the Texas
cases . ^ Jump up to : This takes into account the seven Americans who were medically evacuated from West Africa with Ebola . References ( edit ) Jump up ^ ``
Who are the American Ebola patients ? '' , CNN , October 6 , 2014 Jump up ^ `` Nebraska Hospital : Surgeon With Ebola Has Died '' . ABC News . November 17 ,
2014 . Retrieved November 17 , 2014 . Jump up ^ `` 2014 Ebola Outbreak - Case Count '' . CDC . December 15 , 2014 . Retrieved December 16 , 2014 . Jump up ^ ``
First diagnosed case of Ebola in the U.S. '' CNN .
</item 3>

<item 4>
Ebola virus cases in the United States
Ebola virus cases in the United States - wikipedia Ebola virus cases in the United States Jump to : navigation , search This article is about Ebola virus cases
in the United States starting in 2014 . For the 2013 - 2016 outbreak in Africa , see Ebola virus epidemic in West Africa . For the U.S. discovery of a different
species of the Ebolavirus genus in 1989 , see Reston virus . Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases
contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola
'' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first
was reported in September 2014 .
</item 4>

<item 5>
Ebola virus cases in the United States
CBS News . November 13 , 2014 . Jump up ^ Sieff , Kevin ( November 17 , 2014 ) . `` A doctor 's mistaken Ebola test : ' We were celebrating ... Then everything
fell apart ' '' . Washington Post . Jump up ^ `` Martin Salia , Surgeon With Ebola , Arrives in Nebraska From Sierra Leone '' . New York Times . November 15 ,
2014 . Retrieved November 16 , 2014 . Jump up ^ Goodnough , Abby ; Trenchard , Tommy ( November 17 , 2014 ) . `` Doctor Being Treated for Ebola in Omaha Dies ''
. The New York Times . Jump up ^ U.S.
</item 5>

<item 6>
Ebola virus cases in the United States
Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those
nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered . On September 30
, 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national
visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at
Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 ,
Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were : October 11 , 2014 , a nurse , Nina Pham , who had provided
care to Duncan at the hospital . October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan .
</item 6>

<item 7>
Ebola virus cases in the United States
In addition , on September 21 , a CDC employee , who showed no symptoms of the disease , was flown back to the United States as a preventative measure after low
- risk exposure with a healthcare worker ; he posed no risks to his family or the United States , and he is not known to have died . On September 28 , a fourth
American doctor was admitted to National Institutes of Health hospital . On October 2 , NBC News photojournalist Ashoka Mukpo , covering the outbreak in Liberia
, tested positive for Ebola after showing symptoms . Four other members of the NBC team , including physician Nancy Snyderman , were being closely monitored for
symptoms . Mukpo was evacuated on October 6 to the University of Nebraska Medical Center for treatment in their isolation unit . On October 21 , Mukpo was
declared Ebola - free and allowed to return to his home in Rhode Island . A government official in Sierra Leone announced on November 13 , that a doctor from
Sierra Leone , a permanent resident of the United States married to a U.S. citizen , would be transported to the Nebraska Medical Center for treatment for Ebola
. The doctor , identified later as Dr.
</item 7>

<item 8>
Ebola virus cases in the United States
October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan . October 23 , 2014 , physician Craig Spencer , diagnosed in New York City ; he had just
returned from working with Doctors Without Borders in Guinea , a country in west Africa . Hundreds of people were tested or monitored for potential Ebola virus
infection , but the two nurses were the only confirmed cases of locally transmitted Ebola . Public health experts and the Obama administration opposed
instituting a travel ban on Ebola endemic areas , stating that it would be ineffective and would paradoxically worsen the situation . No one both contracted and
died of Ebola virus disease while in the United States . No new cases have been diagnosed in the United States since Craig Spencer was released from a New York
City hospital on November 11 , 2014 . Contents ( hide ) 1 Cases diagnosed in U.S. 1.1 First case : Thomas Eric Duncan 1.1. 1 Thomas Eric Duncan in Liberia 1.1.
2 Duncan 's illness in Dallas 1.1. 3 Contact tracing 1.1.
</item 8>

<item 9>
Ebola virus cases in the United States
The next day , October 3 , Duncan again reported feeling abdominal pain . That evening , the hospital contacted Chimerix , a biotechnology company developing
Brincidofovir to combat the disease . The next day , Duncan 's organs were failing , and he was intubated to help him breathe . In the afternoon , the hospital
began administering Brincidofovir . Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock . On October 7 , the hospital reported
that Duncan 's condition was improving . However , Duncan died at 7 : 51 a.m. on October 8 , becoming the first person to die in the United States of Ebola
virus disease and the index patient for the later infections of nurses Pham and Vinson . Contact tracing ( edit ) On October 5 , the CDC announced it had lost
track of a homeless man who had been in the same ambulance as Duncan . They announced efforts were underway to find the man and place him in a comfortable and
compassionate monitoring environment . Later that day , the CDC announced that the man had been found and was being monitored .
</item 9>

<item 10>
Ebola virus cases in the United States
WYFF4 . Retrieved October 3 , 2014 . Jump up ^ `` First US Ebola case Thomas Duncan ' critical ' '' . BBC News . October 4 , 2014 . Retrieved October 4 , 2014 .
Jump up ^ `` Texas Ebola patient dies '' . CNN . October 8 , 2014 . Retrieved October 8 , 2014 . ^ Jump up to : Fernandez , Manny ( October 12 , 2014 ) . ``
Texas Health Worker Tests Positive for Ebola '' . New York Times . Retrieved October 12 , 2014 . Jump up ^ `` Ebola outbreak : Second Texas health worker '
tests positive ' '' . BBC News . October 15 , 2014 . Retrieved October 15 , 2014 . ^ Jump up to : `` New York Doctor Just Back From Africa Has Ebola '' . NBC
News . October 23 , 2014 . Retrieved October 23 , 2014 . Jump up ^ Dan Diamond ( October 20 , 2014 ) .
</item 10>
=== END ITEM id=62b123ae873306de ===
