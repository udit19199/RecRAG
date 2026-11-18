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

This file has 7 items.

=== ITEM id=07c0b9d7d166b5b0 ===
Gold sentences and facts:
Sentence 0: Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st Emmanuel Amunike Sporting CP 48 2nd George Weah Paris Saint - Germain 44 Rashidi Yekini Olympiacos 1995 1st George Weah Milan -- 2nd Emmanuel Amunike Sporting CP -- 3rd Daniel Amokachi Everton -- 1st Nwankwo Kanu Internazionale -- 2nd George Weah Milan -- 3rd Daniel Amokachi Beşiktaş -- 1st Victor Ikpeba Monaco 56 2nd Japhet N'Doram Monaco 40 3rd Taribo West Internazionale 35 1998 1st Mustapha Hadji Deportivo La Coruña 76 2nd Jay - Jay Okocha Paris Saint - Germain 74 3rd Sunday Oliseh Ajax 58 1999 1st Nwankwo Kanu Arsenal 46 2nd Samuel Kuffour Bayern Munich 44 3rd Ibrahima Bakayoko Marseille 42 2000 1st Patrick M'Boma Parma 123 2nd Lauren Mallorca 36 3rd Samuel Eto'o Mallorca 29 1st El Hadji Diouf Lens 93 2nd Samuel Kuffour Bayern Munich 66 3rd Samuel Eto'o Mallorca 34 2002 1st El Hadji Diouf Liverpool 93 2nd Papa Bouba Diop Lens 46 3rd Mido Ajax 42 2003 1st Samuel Eto'o Mallorca -- 2nd Didier Drogba Marseille -- 3rd Jay - Jay Okocha Bolton Wanderers -- 1st Samuel Eto'o Barcelona 116 2nd Didier Drogba Chelsea 90 3rd Jay - Jay Okocha Bolton Wanderers 68 2005 1st Samuel Eto'o Barcelona 108 2nd Didier Drogba Chelsea 106 3rd Michael Essien Chelsea 50 2006 1st Didier Drogba Chelsea 79 2nd Samuel Eto'o Barcelona 76 3rd Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74 2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea 43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311
  Fact 0.0: In 1992, Abedi Pele of Marseille ranked 1st with no points listed.
  Fact 0.1: In 1993, Rashidi Yekini of Vitória de Setúbal ranked 1st with no points listed.
  Fact 0.2: In 1994, Emmanuel Amunike of Sporting CP ranked 1st with 48 points.
  Fact 0.3: In 1994, George Weah of Paris Saint-Germain ranked 2nd with 44 points.
  Fact 0.4: In 1994, Rashidi Yekini of Olympiacos is listed with no rank shown.
  Fact 0.5: In 1995, George Weah of Milan ranked 1st with no points listed.
  Fact 0.6: In 1995, Emmanuel Amunike of Sporting CP ranked 2nd with no points listed.
  Fact 0.7: In 1995, Daniel Amokachi of Everton ranked 3rd with no points listed.
  Fact 0.8: In an entry with no year label, Nwankwo Kanu of Internazionale ranked 1st with no points listed.
  Fact 0.9: In an entry with no year label, George Weah of Milan ranked 2nd with no points listed.
  Fact 0.10: In an entry with no year label, Daniel Amokachi of Beşiktaş ranked 3rd with no points listed.
  Fact 0.11: In an entry with no year label, Victor Ikpeba of Monaco ranked 1st with 56 points.
  Fact 0.12: In an entry with no year label, Japhet N'Doram of Monaco ranked 2nd with 40 points.
  Fact 0.13: In an entry with no year label, Taribo West of Internazionale ranked 3rd with 35 points.
  Fact 0.14: In 1998, Mustapha Hadji of Deportivo La Coruña ranked 1st with 76 points.
  Fact 0.15: In 1998, Jay-Jay Okocha of Paris Saint-Germain ranked 2nd with 74 points.
  Fact 0.16: In 1998, Sunday Oliseh of Ajax ranked 3rd with 58 points.
  Fact 0.17: In 1999, Nwankwo Kanu of Arsenal ranked 1st with 46 points.
  Fact 0.18: In 1999, Samuel Kuffour of Bayern Munich ranked 2nd with 44 points.
  Fact 0.19: In 1999, Ibrahima Bakayoko of Marseille ranked 3rd with 42 points.
  Fact 0.20: In 2000, Patrick M'Boma of Parma ranked 1st with 123 points.
  Fact 0.21: In 2000, Lauren of Mallorca ranked 2nd with 36 points.
  Fact 0.22: In 2000, Samuel Eto'o of Mallorca ranked 3rd with 29 points.
  Fact 0.23: In an entry with no year label, El Hadji Diouf of Lens ranked 1st with 93 points.
  Fact 0.24: In an entry with no year label, Samuel Kuffour of Bayern Munich ranked 2nd with 66 points.
  Fact 0.25: In an entry with no year label, Samuel Eto'o of Mallorca ranked 3rd with 34 points.
  Fact 0.26: In 2002, El Hadji Diouf of Liverpool ranked 1st with 93 points.
  Fact 0.27: In 2002, Papa Bouba Diop of Lens ranked 2nd with 46 points.
  Fact 0.28: In 2002, Mido of Ajax ranked 3rd with 42 points.
  Fact 0.29: In 2003, Samuel Eto'o of Mallorca ranked 1st with no points listed.
  Fact 0.30: In 2003, Didier Drogba of Marseille ranked 2nd with no points listed.
  Fact 0.31: In 2003, Jay-Jay Okocha of Bolton Wanderers ranked 3rd with no points listed.
  Fact 0.32: In an entry with no year label, Samuel Eto'o of Barcelona ranked 1st with 116 points.
  Fact 0.33: In an entry with no year label, Didier Drogba of Chelsea ranked 2nd with 90 points.
  Fact 0.34: In an entry with no year label, Jay-Jay Okocha of Bolton Wanderers ranked 3rd with 68 points.
  Fact 0.35: In 2005, Samuel Eto'o of Barcelona ranked 1st with 108 points.
  Fact 0.36: In 2005, Didier Drogba of Chelsea ranked 2nd with 106 points.
  Fact 0.37: In 2005, Michael Essien of Chelsea ranked 3rd with 50 points.
  Fact 0.38: In 2006, Didier Drogba of Chelsea ranked 1st with 79 points.
  Fact 0.39: In 2006, Samuel Eto'o of Barcelona ranked 2nd with 76 points.
  Fact 0.40: In 2006, Michael Essien of Chelsea ranked 3rd with 36 points.
  Fact 0.41: In 2007, Frédéric Kanouté of Sevilla ranked 1st with no points listed.
  Fact 0.42: In 2007, Michael Essien of Chelsea ranked 2nd with no points listed.
  Fact 0.43: In 2007, Didier Drogba of Chelsea ranked 3rd with no points listed.
  Fact 0.44: In 2008, Emmanuel Adebayor of Arsenal ranked 1st with 74 points.
  Fact 0.45: In 2008, Mohamed Aboutrika of Al-Ahly ranked 2nd with 53 points.
  Fact 0.46: In 2008, Michael Essien of Chelsea ranked 3rd with 36 points.
  Fact 0.47: In 2009, Didier Drogba of Chelsea ranked 1st with 92 points.
  Fact 0.48: In 2009, Samuel Eto'o of Internazionale ranked 2nd with 69 points.
  Fact 0.49: In 2009, Michael Essien of Chelsea ranked 3rd with 43 points.
  Fact 0.50: In an entry with no year label, Samuel Eto'o of Internazionale ranked 1st with no points listed.
  Fact 0.51: In an entry with no year label, Asamoah Gyan of Sunderland ranked 2nd with no points listed.
  Fact 0.52: In an entry with no year label, Didier Drogba of Chelsea ranked 3rd with no points listed.
  Fact 0.53: In 2011, Yaya Touré of Manchester City ranked 1st with no points listed.
  Fact 0.54: In 2011, Seydou Keita of Barcelona ranked 2nd with no points listed.
  Fact 0.55: In 2011, André Ayew of Marseille ranked 3rd with no points listed.
  Fact 0.56: In 2012, Yaya Touré of Manchester City ranked 1st with no points listed.
  Fact 0.57: In 2012, Didier Drogba of Shanghai Shenhua ranked 2nd with no points listed.
  Fact 0.58: In 2012, Alex Song of Barcelona ranked 3rd with no points listed.
  Fact 0.59: In 2013, Yaya Touré of Manchester City ranked 1st with 373 points.
  Fact 0.60: In 2013, John Obi Mikel of Chelsea ranked 2nd with 265 points.
  Fact 0.61: In 2013, Didier Drogba of Galatasaray ranked 3rd with 236 points.
  Fact 0.62: In 2014, Yaya Touré of Manchester City ranked 1st with 175 points.
  Fact 0.63: In 2014, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 2nd with 120 points.
  Fact 0.64: In 2014, Vincent Enyeama of Lille ranked 3rd with 105 points.
  Fact 0.65: In 2015, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 1st with 143 points.
  Fact 0.66: In 2015, Yaya Touré of Manchester City ranked 2nd with 136 points.
  Fact 0.67: In 2015, André Ayew of Swansea City ranked 3rd with 112 points.
  Fact 0.68: In 2016, Riyad Mahrez of Leicester City ranked 1st with 361 points.
  Fact 0.69: In 2016, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 2nd with 313 points.
  Fact 0.70: In 2016, Sadio Mané of Liverpool ranked 3rd with 186 points.
  Fact 0.71: In 2017, Mohamed Salah of Liverpool ranked 1st with 625 points.
  Fact 0.72: In 2017, Sadio Mané of Liverpool ranked 2nd with 507 points.
  Fact 0.73: In 2017, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 3rd with 311 points.

Retrieved items:
<item 1>
African Footballer of the Year
It had already been replaced by an official award given out by the Confederation of African Football since 1992 . Year Rank Player Team Points 1970 1st Salif
Keïta Saint - Étienne 54 2nd Laurent Pokou ASEC Abidjan 28 Ali Abo Greisha Ismaily 1971 1st Ibrahim Sunday Asante Kotoko 29 2nd Robert Mensah Asante Kotoko 15
3rd François Ndoumbé Canon Yaoundé 13 1972 1st Chérif Souleymane Hafia 21 2nd Bwanga Tshimen TP Mazembe 16 3rd Petit Sory Hafia 14 1973 1st Bwanga Tshimen TP
Mazembe 49 2nd Kazadi Mwamba TP Mazembe 44 3rd Laurent Pokou ASEC Abidjan 41 1st Paul Moukila CARA Brazzaville 57 2nd Lobilo Boba Vita Club 32 3rd Hassan
Shehata Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd
</item 1>

<item 2>
African Footballer of the Year
Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama
Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez
Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané
Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311 Wins By Country ( edit ) Nation Winners Runners - up Third places Cameroon 11 8 11 Ivory
Coast 8 6 7 Ghana 6 7 7 Nigeria 5 6 7 Morocco 0 Algeria 5 Liberia -- Egypt Mali -- Senegal Guinea DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- --
Chad -- -- Wins
</item 2>

<item 3>
African Footballer of the Year
Cannes 52 1992 1st Abedi Pele Marseille 198 2nd George Weah Paris Saint - Germain 161 3rd Tony Yeboah Eintracht Frankfurt 64 1993 1st Abedi Pele Marseille 119
2nd Tony Yeboah Eintracht Frankfurt 117 3rd Rashidi Yekini Vitória de Setúbal 104 1994 1st George Weah Paris Saint - Germain 148 2nd Emmanuel Amunike Sporting
CP 133 3rd Daniel Amokachi Everton 99 CAF award ( 1992 -- present ) ( edit ) Samuel Eto'o won the award four times . Yaya Touré won the award four consecutive
times from 2011 to 2014 . Pierre - Emerick Aubameyang was the first player from Gabon to win the award in 2015 . Mohamed Salah is the latest player to win the
award in 2017 . Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st
</item 3>

<item 4>
African Footballer of the Year
Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2013 - 07 - 02 . Jump
up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2016 - 09 - 29 .
</item 4>

<item 5>
African Footballer of the Year
Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd Roger Milla Tonnerre Yaoundé 24 Mamadou Aliou Kéïta Hafia 1976 1st Roger Milla Tonnerre Yaoundé 33 2nd Papa Camara
Hafia 32 3rd Ali Bencheikh MC Alger 27 1977 1st Tarak Dhiab Espérance Tunis 45 2nd Papa Camara Hafia 33 3rd Segun Odegbami Shooting Stars 29 1978 1st Karim
Abdul Razak Asante Kotoko 58 2nd Ali Bencheikh MC Alger 33 3rd Thomas N'Kono Canon Yaoundé 29 1979 1st Thomas N'Kono Canon Yaoundé 55 2nd Adolf Armah Hearts of
Oak 23 3rd Kerfalla Bangoura Horoya 15 1980 1st Jean Manga - Onguéné Canon Yaoundé 64 2nd Segun Odegbami Shooting Stars 41 3rd Théophile Abega Canon Yaoundé
</item 5>

<item 6>
Graph query row:
award_winner: Yaya Touré
award: African Footballer of the Year 2014
year: 2014
</item 6>

<item 7>
African Footballer of the Year
Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74
2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea
43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita
Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya
Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014
</item 7>

<item 8>
African Footballer of the Year
African Footballer of the Year - wikipedia African Footballer of the Year Jump to : navigation , search The African Footballer of the Year award , presented to
the best African footballer each year , has been conferred by the Confederation of African Football ( CAF ) since 1992 . An earlier African Footballer of the
Year Golden Ball award was given out between 1970 and 1994 by France Football magazine . The changes resulted in parallel Golden Ball awards given out to Abedi
Pele and George Weah in 1993 and 1994 by the magazine although the CAF sponsored awards for those years were won respectively by Rashidi Yekini and Emmanuel
Amuneke , as well as two awards given to Abedi Pelé in 1992 . France Football discontinued the election from 1995 after the European Ballon d'Or -- also awarded
by the magazine -- had been opened to all players in the European leagues . On 2 February 2008 , Sevilla 's Frédéric Kanouté was named the 2007 African
Footballer of the Year .
</item 8>

<item 9>
African Footballer of the Year
Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC Mulhouse -- -- Hearts of Oak -- -- FC Sion -- -- Aydınspor -- -- JS Kabylie -- --
Bolton Wanderers -- -- Everton FC -- -- Ajax Amsterdam -- -- Ismaily SC -- -- Zamalek SC -- AS Cannes -- -- Stade Lavallois -- -- Lille OSC -- -- Stade Rennais
-- -- Olympiacos -- -- Horoya AC -- -- PSV Eindhoven -- -- OC Agaza -- -- Beşiktaş -- -- Galatasaray -- -- Swansea City -- -- See also ( edit ) BBC African
Footballer of the Year African Women Footballer of the Year FIFA World Player of the Year CAF Awards References ( edit ) Jump up ^ `` African Player of the Year
1993 '' . Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year 1994 '' . Rsssf.com. 2001 - 01 - 05 .
</item 9>

<item 10>
African Footballer of the Year
DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- -- Chad -- -- Wins By Club ( edit ) Club Winners Runners - up Third places Olympique Marseille
Manchester City -- Chelsea 6 AS Monaco Barcelona Asante Kotoko -- Canon Yaoundé RCD Mallorca Inter Milan Liverpool Arsenal -- -- Paris SG -- Hafia Conacry TP
Mazembe -- Borussia Dortmund -- Sporting CP -- Al Ahly FC Porto Tonnerre Yaoundé -- RC Lens -- Milan -- Sporting CP -- GC Mascara -- Vitória de Setúbal --
Cercle Brugge -- -- CARA Brazzaville -- -- Leicester City -- -- AS Saint - Étienne -- -- Toulouse FC -- -- Parma -- -- FAR Rabat -- -- SCC Mohammédia -- -- JS
Saint - Pierroise -- -- Deportivo de La Coruña -- -- Espanyol -- -- Sevilla -- -- Espérance ST -- -- Bayern Munich -- -- MC Alger -- ASEC Mimosas -- Montpellier
-- Eintracht Frankfurt -- Shooting Stars -- Shanghai Shenhua -- -- Vita Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC
</item 10>
=== END ITEM id=07c0b9d7d166b5b0 ===

=== ITEM id=64a7077da2bee01e ===
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
University of San Diego School of Law President of the Court of Appeal of Sri Lanka Andrew Somawansa North Western 28 May 2006 -- October 2008 -- -- Rajapaksa
University of Ceylon University of Nevada President of the Court of Appeal of Sri Lanka Jagath Balapatabendi Southern -- -- -- Rajapaksa Ceylon Law College
Judge of the Court of Appeal Kanagasabapathy Sripavan † 1952 -- 27 March 2008 -- present 30 January 2015 -- 28 February 2017 7003368200000000000 ♠ 3,682
7003118200000000000 ♠ 1,182 Rajapaksa Sri Lanka Law College Judge of the Court of Appeal P.A. Ratnayake 2 May 2008 -- 15 February 2013 -- 7003175100000000000 ♠
1,751 -- Rajapaksa University of Brussels Chandra Ekanayake November 2008 -- 2016 -- -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri
Lanka S.I.
</item 1>

<item 2>
List of Justices of the Supreme Court of Sri Lanka
760 2,554 Kotelawala 98 Thusew Samuel Fernando 2 May 1956 -- -- -- Bandaranaike Colombo Law College Solicitor General of Sri Lanka 99 Nadaraja Sinnetamby 3 May
1956 -- -- -- Bandaranaike 100 Lionel de Silva 5 November 1959 -- -- -- Dahanayake 101 Leonard de Silva 16 August 1966 -- -- -- Senanayake 102 Henry Thambiah
Northern 1906 -- 1997 1 December 1961 -- 27 May 1968 -- 2,369 -- Bandaranaike University College , Colombo Ceylon Law College University of London Commissioner
of Assize 103 Kingsley Herath 25 January 1962 -- -- -- Bandaranaike 104 Asoka Abeyesundere Southern 1906 - 1988 20 April 1962 -- 3 May 1968 -- 2,
</item 2>

<item 3>
List of Justices of the Supreme Court of Sri Lanka
450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri
Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College
University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne †
30 January 2015 -- present -- 7003118200000000000 ♠ 1,
</item 3>

<item 4>
List of Justices of the Supreme Court of Sri Lanka
9 January - 5 June 1888 ) Jump up ^ On leave ( 25 November 1890 - February 1891 ) Jump up ^ Acting Justice ( 12 May 1903 -- 28 November 1910 ) References ( edit
) Jump up ^ Amerasinghe , A. Ranjit B ( 1986 ) , The Supreme Court of Sri Lanka : the first 185 years , Sarvodaya Book Pub . Services , ISBN 978 - 955 - 599 -
000 - 4 Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^
Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most
recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update
. Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update .
</item 4>

<item 5>
List of Justices of the Supreme Court of Sri Lanka
Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya
Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 --
present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of
Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,
</item 5>

<item 6>
List of Justices of the Supreme Court of Sri Lanka
Kulatunge -- -- -- G.P.S. de Silva -- 1991 -- 1999 Sarath N. Silva 1995 -- 16 September 1999 16 September 1999 -- 7 June 2009 3,552 Kumaratunga Sri Lanka Law
College University of Brussels President of the Court of Appeal of Sri Lanka Shirani Bandaranayake North Western 1958 - 30 October 1996 -- 13 January 2013 28
January 2015 -- 29 January 2015 18 May 2011 -- 13 January 2013 28 January 2015 -- 29 January 2015 5,919 + 1 606 + 1 Kumaratunga University of Colombo University
of London Lawyer C.V. Vigneswaran Western 1939 - March 2001 -- October 2004 -- -- Kumaratunga University of Ceylon Ceylon Law College Judge of the Court of
Appeal Asoka de Silva 1946 - 1 August 2001 -- 17 May 2011 8 June 2009 -- 17 May 2011 3,
</item 6>

<item 7>
List of Justices of the Supreme Court of Sri Lanka
576 708 Kumaratunga University of Peradeniya University of Colombo University of Illinois at Urbana -- Champaign President of the Court of Appeal of Sri Lanka
Nihal Jayasinghe 19 February 2003 -- 29 February 2008 -- 1,836 -- Kumaratunga University of Colombo Judge of the Court of Appeal Nimal Dissanayake -- 23 April
2008 -- -- Kumaratunga Shiranee Tilakawardane 19 February 2003 - 5 April 2014 -- 1,901 -- Kumaratunga Sri Lanka Law College President of the Court of Appeal of
Sri Lanka Nimal Gamini Amaratunga January 2005 -- March 2014 -- -- Kumaratunga University of Colombo International Maritime Law Institute Judge of the Court of
Appeal Saleem Marsoof February 2005 -- 2014 -- -- Kumaratunga University of Colombo ,
</item 7>

<item 8>
List of Justices of the Supreme Court of Sri Lanka
List of justices of the Supreme Court of Sri Lanka - wikipedia List of justices of the Supreme Court of Sri Lanka Jump to : navigation , search Shirani
Bandaranayake was the 43rd Chief Justice of Sri Lanka and the first and only woman to hold the position . List of Justices of the Supreme Court of Sri Lanka In
this list , justices are placed in the order in which they took the judicial oath of office and thereby started their term of office . While many of the
justices ' positions prior to appointment are simply listed as `` lawyer '' , many had part - time positions , such as teaching , or acted as counsel to various
levels of government . Contents ( hide ) 1 Justices 2 See also 3 Notes 4 References 5 Bibliography Justices ( edit ) † Denotes current justices Began as
Provisional or Acting Puisne Justice ♠ Tenure demoted to de facto † Died in office No .
</item 8>

<item 9>
List of Justices of the Supreme Court of Sri Lanka
182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present --
7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 --
present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠
679 -- Sirisena University of Colombo Judge of the Court of Appeal See also ( edit ) List of Chief Justices of Sri Lanka List of Justices of the Supreme Court
of Sri Lanka by court composition Notes ( edit ) Jump up ^ Provisional Chief Justice Jump up ^ On leave ( 1 January - 28 April 1871 ) Jump up ^ Acting Chief
Justice ( 1 - 10 October 1879 ,
</item 9>

<item 10>
List of Justices of the Supreme Court of Sri Lanka
Jump up ^ Current to most recent page update . Bibliography ( edit ) Amerasinghe , A.R.B. ( 1986 ) . The Supreme Court of Sri Lanka : the first 185 years .
Ratmalana , Sri Lanka : Sarvodaya Book Pub . Services . ISBN 955599000X . ( hide ) Supreme Court of Sri Lanka The Court Decisions Procedure History Court
Building Current membership Chief Justice Kanagasabapathy Sripavan Puisne Justices Priyasath Dep Eva Wansundera Buwaneka Aluwihare Sisira De Abrew Priyantha
Jayawardena Upaly Abeyratne Anil Goonaratne K.T.
</item 10>
=== END ITEM id=64a7077da2bee01e ===

=== ITEM id=e73c64f06d1ab2c3 ===
Gold sentences and facts:
Sentence 0: As a technique for drawing , the closest predecesor to the pencil was Silverpoint until in 1565 ( some sources say as early as 1500 ) , a large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite in Borrowdale parish , Cumbria , England . This particular deposit of graphite was extremely pure and solid , and it could easily be sawn into sticks . It remains the only large - scale deposit of graphite ever found in this solid form . Chemistry was in its infancy and the substance was thought to be a form of lead . Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally mean lead pen .
  Fact 0.0: Before the pencil, the closest predecessor as a drawing technique was Silverpoint, until 1565 (some sources say as early as 1500).
  Fact 0.1: A large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite.
  Fact 0.2: The Grey Knotts graphite deposit is in Borrowdale parish, Cumbria, England.
  Fact 0.3: The graphite deposit at Grey Knotts was extremely pure and solid.
  Fact 0.4: The graphite deposit at Grey Knotts could easily be sawn into sticks.
  Fact 0.5: The Grey Knotts graphite deposit remains the only large-scale deposit of graphite ever found in this solid form.
  Fact 0.6: Chemistry was in its infancy when the graphite was found.
  Fact 0.7: The substance was thought to be a form of lead.
  Fact 0.8: The substance was called plumbago, Latin for 'lead ore'.
  Fact 0.9: The pencil core is still referred to as 'lead' or 'a lead'.
  Fact 0.10: Many people have the misconception that the graphite in the pencil is lead.
  Fact 0.11: The black core of pencils is still referred to as lead.
  Fact 0.12: The black core of pencils never contained the element lead.
  Fact 0.13: The German word for pencil, bleistift, literally means lead pen.
  Fact 0.14: The Irish word for pencil, peann luaidhe, literally means lead pen.
  Fact 0.15: The Arabic word for pencil, قلم رصاص (qalam raṣāṣ), literally means lead pen.
  Fact 0.16: Some other languages also use words for pencil that literally mean lead pen.

Retrieved items:
<item 1>
Pencil
Their concept involved the hollowing out of a stick of juniper wood . Shortly thereafter , a superior technique was discovered : two wooden halves were carved ,
a graphite stick inserted , and the halves then glued together -- essentially the same method in use to this day . New pencils from graphite powder , and
graphite and clay The first attempt to manufacture graphite sticks from powdered graphite was in Nuremberg , Germany , in 1662 . It used a mixture of graphite ,
sulphur , and antimony . English and German pencils were not available to the French during the Napoleonic Wars ; France , under naval blockade imposed by Great
Britain , was unable to import the pure graphite sticks from the British Grey Knotts mines -- the only known source in the world . France was also unable to
import the inferior German graphite pencil substitute . It took the efforts of an officer in Napoleon 's army to change this . In 1795 , Nicolas - Jacques Conté
discovered a method of mixing powdered graphite with clay and forming the mixture into rods that were then fired in a kiln . By varying the ratio of graphite to
clay , the hardness of the graphite rod could also be varied .
</item 1>

<item 2>
Pencil
Contents ( hide ) 1 History 1.1 Camel 's hair pencil 1.2 Discovery of graphite deposit 1.3 Wood holders added 1.4 New pencils from graphite powder , and
graphite and clay 1.5 The pencil in America 1.6 Eraser attached 1.7 Pencil extenders 2 Types 2.1 By marking material 2.2 By use 2.3 By shape 2.4 By size 2.5 By
manufacture 3 Health 4 Manufacture 5 Grading and classification 5.1 Pencil hardness test 6 External colour and shape of pencils 7 Notable pencil users 8
Manufacturers 9 See also 10 References 11 Bibliography 12 Further reading 13 External links History Old Soviet colored pencils with box ( circa 1959 ) Camel 's
hair pencil Pencil , from Old French pincel , from Latin penicillus a `` little tail '' ( see penis ; pincellus is Latin from the post-classical period )
originally referred to an artist 's fine brush of camel hair , also used for writing before modern lead or chalk pencils .
</item 2>

<item 3>
Pencil
Because graphite is soft , it requires some form of encasement . Graphite sticks were initially wrapped in string or sheepskin for stability . England would
enjoy a monopoly on the production of pencils until a method of reconstituting the graphite powder was found in 1662 in Italy . However , the distinctively
square English pencils continued to be made with sticks cut from natural graphite into the 1860s . The town of Keswick , near the original findings of block
graphite , still manufactures pencils , the factory also being the location of the Cumberland Pencil Museum . The meaning of `` graphite writing implement ''
apparently evolved late in the 16th century . Wood holders added Palomino Blackwing 602 pencils Around 1560 , an Italian couple named Simonio and Lyndiana
Bernacotti made what are likely the first blueprints for the modern , wood - encased carpentry pencil . Their version was a flat , oval , more compact type of
pencil . Their concept involved the hollowing out of a stick of juniper wood .
</item 3>

<item 4>
Pencil
Archived from the original on 5 March 2016 . Retrieved 5 October 2015 . Jump up ^ Bennett , Howard J. ( 26 November 2014 ) . `` Ever wondered about the lead in
pencils ? '' . The Washington Post . ISSN 0190 - 8286 . Archived from the original on 6 November 2015 . Retrieved 5 October 2015 . Jump up ^ `` Pencil
swallowing : MedlinePlus Medical Encyclopedia '' . www.nlm.nih.gov . Archived from the original on 6 October 2015 . Retrieved 5 October 2015 . Jump up ^ ``
graphite pencils The Weekend Historian '' . umeshmadan.wordpress.com . Archived from the original on 7 October 2015 . Retrieved 5 October 2015 . Jump up ^ ``
Origins of the Pencil ( Circa 1500 -- 1565 ) : HistoryofInformation.com '' . www.historyofinformation.com . Retrieved 2016 - 10 - 29 .
</item 4>

<item 5>
Pencil
Though the archetypal pencil was an artist 's brush , the stylus , a thin metal stick used for scratching in papyrus or wax tablets , was used extensively by
the Romans and for palm - leaf manuscripts . Discovery of graphite deposit As a technique for drawing , the closest predecesor to the pencil was Silverpoint
until in 1565 ( some sources say as early as 1500 ) , a large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite in
Borrowdale parish , Cumbria , England . This particular deposit of graphite was extremely pure and solid , and it could easily be sawn into sticks . It remains
the only large - scale deposit of graphite ever found in this solid form . Chemistry was in its infancy and the substance was thought to be a form of lead .
Consequently , it was called plumbago ( Latin for `` lead ore '' ) .
</item 5>

<item 6>
Pencil
This method of manufacture , which had been earlier discovered by the Austrian Joseph Hardtmuth , the founder of the Koh - I - Noor in 1790 , remains in use .
In 1802 , the production of graphite leads from graphite and clay was patented by the Koh - I - Noor company in Vienna . In England , pencils continued to be
made from whole sawn graphite . Henry Bessemer 's first successful invention ( 1838 ) was a method of compressing graphite powder into solid graphite thus
allowing the waste from sawing to be reused . The pencil in america Pencil , perhaps made by Henry David Thoreau , in the Concord Museum Pencil manufacturing .
The top sequence shows the old method that required pieces of graphite to be cut to size ; the lower sequence is the new , current method using rods of graphite
and clay . American colonists imported pencils from Europe until after the American Revolution . Benjamin Franklin advertised pencils for sale in his
Pennsylvania Gazette in 1729 , and George Washington used a three - inch pencil when he surveyed the Ohio Territory in 1762 .
</item 6>

<item 7>
Pencil
Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people
have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the
element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally
mean lead pen . The value of graphite would soon be realised to be enormous , mainly because it could be used to line the moulds for cannonballs ; the mines
were taken over by the Crown and were guarded . When sufficient stores of graphite had been accumulated , the mines were flooded to prevent theft until more was
required . The usefulness of graphite for pencils was discovered as well , but graphite for pencils had to be smuggled . The news of the usefulness of these
early pencils spread far and wide , attracting the attention of artists all over the known world . Because graphite is soft , it requires some form of
encasement .
</item 7>

<item 8>
Pencil
Manufacture The lead of the pencil is a mix of finely ground graphite and clay powders . Before the two substances are mixed , they are separately cleaned of
foreign matter and dried in a manner that creates large square cakes . Once the cakes have fully dried , the graphite and the clay squares are mixed together
using water . The amount of clay content added to the graphite depends on the intended pencil hardness ( lower proportions of clay makes the core softer ) , and
the amount of time spent on grinding the mixture determines the quality of the lead . The mixture is then shaped into long spaghetti - like strings ,
straightened , dried , cut , and then tempered in a kiln . The resulting strings are dipped in oil or molten wax , which seeps into the tiny holes of the
material and allows for the smooth writing ability of the pencil . A juniper or incense - cedar plank with several long parallel grooves is cut to fashion a ``
slat , '' and the graphite / clay strings are inserted into the grooves . Another grooved plank is glued on top , and the whole assembly is then cut into
individual pencils , which are then varnished or painted .
</item 8>

<item 9>
Pencil
www.historyofinformation.com . Retrieved 2016 - 10 - 29 . Jump up ^ `` Lead Facts - Uses , Properties , Element Pb , Plumbing , Pipes , Weights '' .
www.sciencekids.co.nz . Archived from the original on 30 October 2016 . Retrieved 29 October 2016 . Jump up ^ `` Keswick Pencil Museum '' . Pencilmuseum.co.uk .
Archived from the original on 3 August 2009 . Retrieved 23 July 2009 . Jump up ^ Douglas Harper ( 27 June 2012 ) . `` pencil '' . Online Etymology Dictionary .
Archived from the original on 23 August 2012 . Retrieved 27 June 2012 . Jump up ^ `` Who invented the pencil ? '' . Rocket City Space Pioneers . Archived from
the original on 2011 - 10 - 22 . Jump up ^ `` Timeline Outline ViewEra : 1500 - 1550 '' . Historyofscience.com . Retrieved 18 August 2012 . Jump up ^ `` Pencil
'' .
</item 9>

<item 10>
Pencil
p. 419 . Archived from the original on 17 January 2018 . Jump up ^ Steven S. Zumdahl & Susan A. Zumdahl ( 2008 ) . `` No lead pencils '' . Chemistry . Belmont ,
CA : Cengage Learning . p. 343 . ISBN 0 - 547 - 12532 - 1 . Jump up ^ Martin and Jean Norgate , Geography Department , Portsmouth University ( 2008 ) . `` Old
Cumbria Gazetteer , black lead mine , Seathwaite '' . Archived from the original on 7 January 2009 . Retrieved 19 May 2008 . CS1 maint : Multiple names :
authors list ( link ) Jump up ^ Alfred Wainwright ( 2005 ) . A Pictorial Guide to the Lakeland Fells , Western Fells . ISBN 0 - 7112 - 2460 - 9 . Jump up ^ ``
Graphite from the Plumbago Mine , Borrowdale , England '' . Department of Physics at Michigan Technological University .
</item 10>
=== END ITEM id=e73c64f06d1ab2c3 ===

=== ITEM id=443959ed30540c82 ===
Gold sentences and facts:
Sentence 0: Name Age Hometown Boarded Destination Lifeboat Body Allen , Miss Elizabeth Walton 29 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Allison , Mr. Hudson Joshua Creighton 30 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 135 and chauffeur , Mr. George Swane , 19 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 294 and cook , Miss Amelia Mary `` Mildred '' Brown 18 London , England , UK Southampton Montreal , Quebec , Canada 11 Allison , Mrs. Bessie Waldo ( née Daniels ) 25 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada and maid , Miss Sarah Daniels 33 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 8 Allison , Miss Helen Loraine Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Allison , Master Hudson Trevor 0.11 ! 11 mo . Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 11 and nurse , Miss Alice Catherine Cleaver 22 London , England , UK Southampton Montreal , Quebec , Canada 11 Anderson , Mr. Harry 47 New York City , New York , US Southampton New York City , New York , US Andrews , Miss Kornelia Theodosia 62 Hudson , New York , US Cherbourg Hudson , New York , US 10 Andrews , Mr. Thomas 39 Belfast , Ireland , UK Belfast New York City Appleton , Mrs. Charlotte Lane ( née Lamson ) 53 New York City Southampton New York City Artagaveytia , Mr. Ramon 71 Buenos Aires , Argentina Cherbourg New York City 22 Astor , Colonel John Jacob IV 47 New York City Cherbourg New York City 124 Astor , Mrs. Madeleine Talmage ( née Force ) 18 New York City Cherbourg New York City and maid , Miss Rosalie Bidois 46 New York City Cherbourg New York City and nurse , Miss Caroline Louise Endres 39 New York City Cherbourg New York City Aubart , Mrs. Léontine Pauline 24 Paris , France Cherbourg New York City 9 and maid , Miss Emma Sägesser 24 Paris , France Cherbourg New York City 9 Barkworth , Mr. Algernon Henry Wilson 47 Hessle , East Yorkshire , England , UK Southampton New York City Baumann , Mr. John D . 60 New York City Cherbourg New York City Baxter , Mrs. Hélène ( née de Lanaudière - Chaput ) 50 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada 6 Baxter , Mr. Quigg Edmond 24 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada Beattie , Mr. Thomson 36 Fergus , Ontario , Canada Southampton Fergus , Ontario , Canada 331 Beckwith , Mr. Richard Leonard 37 New York City Southampton New York City 5 Beckwith , Mrs. Sallie ( née Monypeny ) 46 New York City Southampton New York City 5 Behr , Mr. Karl Howell 26 New York City Cherbourg New York City 5 Birnbaum , Mr. Jakob 24 Antwerp , Belgium Cherbourg San Francisco , California , US 148 Bishop , Mr. Dickinson H. `` Dick '' 25 Dowagiac , Michigan , US Cherbourg Dowagiac , Michigan , US 7 Bishop , Mrs. Helen ( née Walton ) 19 Dowagiac , Michigan , US Cherbourg Dowagiac , Michigan , US 7 Björnström - Steffanson , Mr. Mauritz Håkan 28 Stockholm , Sweden Southampton Washington , D.C. , US Blackwell , Mr. Stephen Weart 45 Trenton , New Jersey , US Southampton Trenton , New Jersey , US Blank , Mr. Henry 39 Glen Ridge , New Jersey , US Cherbourg Glen Ridge , New Jersey , US 7 Bonnell , Miss Elizabeth 61 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Bonnell , Miss Caroline 30 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Borebank , Mr. John James 42 London , England , UK Southampton Toronto , Ontario , Canada Bowerman , Miss Elsie Edith 22 St. Leonards - on - Sea , East Sussex , England , UK Southampton New York City 6 Brady , Mr. John Bertram 41 Pomeroy , Washington , US Southampton Pomeroy , Washington , US Brandeis , Mr. Emil 48 Omaha , Nebraska , US Cherbourg Omaha , Nebraska , US 208 Brereton , Mr. George Andrew ( alias George A. Brayton ) 37 Los Angeles , US Southampton Los Angeles , California , US 9 Brewe , Dr. Arthur Jackson 45 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US Brown , Mrs. Caroline Lane ( née Lamson ) 59 Belmont , Massachusetts , US Southampton Belmont , Massachusetts , US Brown , Mrs. Margaret ( née Tobin ) 44 Denver , Colorado , US Cherbourg Denver , Colorado , US 6 Bucknell , Mrs. Emma Eliza ( née Ward ) 59 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 8 and maid , Miss Albina Bazzani 36 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 8 Butt , Major Archibald Willingham 46 Washington , D.C. , US Southampton Washington , D.C. , US Calderhead , Mr. Edward Pennington 42 New York City Southampton New York City 5 Candee , Mrs. Helen Churchill ( née Hungerford ) 53 Washington , D.C. , US Cherbourg Washington , D.C. , US 6 Cardeza , Mrs. Charlotte Wardle ( née Drake ) 58 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US and maid , Miss Annie Moore Ward 38 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US Cardeza , Mr. Thomas Drake Martinez 36 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US and valet , Mr. Gustave J. Lesueur 35 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US Carlsson , Mr. Frans Olof 33 New York City Southampton New York City Carrau , Mr. Francisco M . 31 Montevideo , Uruguay Southampton Montevideo , Uruguay Carrau , Mr. José Pedro 17 Montevideo , Uruguay Southampton Montevideo , Uruguay Carter , Mr. William Ernest 36 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and valet , Mr. Alexander Cairns 28 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and chauffeur , Mr. Charles Augustus Aldworth 30 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Mrs. Lucile ( née Polk ) 36 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and maid , Miss Auguste Serreplan 30 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Miss Lucile Polk 13 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Master William Thornton II 11 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Case , Mr. Howard Brown 49 Ascot , Berkshire , England , UK Southampton Rochester , New York , US Cassebeer , Mrs. Eleanor Genevieve ( née Fosdick ) 36 New York City Cherbourg New York City 5 Cavendish , Mr. Tyrell William 36 London , England , UK Southampton New York City 172 Cavendish , Mrs. Julia Florence ( née Siegel ) 25 London , England , UK Southampton New York City 6 and maid , Miss Ellen `` Nellie '' Barber 26 London , England , UK Southampton New York City 6 Chaffee , Mr. Hubert Fuller 46 Amenia , North Dakota , US Southampton Amenia , North Dakota , US Chaffee , Mrs. Carrie Constance ( née Toogood ) 47 Amenia , North Dakota , US Southampton Amenia , North Dakota , US Chambers , Mr. Norman Campbell 27 New York City Southampton New York City 5 Chambers , Mrs. Bertha ( née Griggs ) 32 New York City Southampton New York City 5 Cherry , Miss Gladys 30 London , England , UK Southampton Vancouver , British Columbia , Canada 8 Chevré , Mr. Paul Romaine Marie Léonce 45 Paris , France Cherbourg Ottawa , Ontario , Canada 7 Chibnall , Mrs. Edith Martha Bowerman ( née Barber ) 48 St. Leonards - on - Sea , East Sussex , England , UK Southampton New York City 6 Chisholm , Mr. Roderick Robert Crispin 40 Belfast , Ireland , UK Belfast New York City Clark , Mr. Walter Miller 27 Los Angeles , California , US Cherbourg Los Angeles , California , US Clark , Mrs. Virginia Estelle ( née McDowell ) 26 Los Angeles , California , US Cherbourg Los Angeles , California , US Clifford , Mr. George Quincy 40 Stoughton , Massachusetts , US Southampton Stoughton , Massachusetts Colley , Mr. Edward Pomeroy 37 Dublin , Ireland , UK Southampton Vancouver , British Columbia , Canada Compton , Mrs. Mary Eliza ( née Ingersoll ) 64 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US 14 Compton , Miss Sara Rebecca 39 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US 14 Compton , Mr. Alexander Taylor Jr . 37 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US Cornell , Mrs. Malvina Helen ( née Lamson ) 55 New York City Southampton New York City Crafton , Mr. John Bertram 59 Roachdale , Indiana , US Southampton Roachdale , Indiana , US Crosby , Captain Edward 70 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 269 Crosby , Mrs. Catherine Elizabeth ( née Halstead ) 64 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 7 Crosby , Miss Harriette Rebecca 39 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 7 Cumings , Mr. John Bradley 39 New York City Cherbourg New York City Cumings , Mrs. Florence Briggs ( née Thayer ) 35 New York City Cherbourg New York , US Daly , Mr. Peter Dennis 51 Lima , Peru Southampton Lima , Peru Daniel , Mr. Robert Williams 27 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Davidson , Mr. Thornton 31 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Davidson , Mrs. Orian ( née Hays ) 27 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Dick , Mr. Albert Adrian 31 Calgary , Alberta , Canada Southampton Calgary , Alberta , Canada Dick , Mrs. Vera ( née Gillespie ) 17 Calgary , Alberta , Canada Southampton Calgary , Alberta , Canada Dineston , Mrs. Nancy May Bowles 28 London , England , UK Southampton New York City , USA Dineston , George Samuel 30 London , England , UK Southampton New York City , USA Dodge , Dr. Washington 52 San Francisco , California , US Southampton San Francisco , California , US 13 Dodge , Mrs. Ruth ( née Vidaver ) 34 San Francisco , California , US Southampton San Francisco , California , US 5 Dodge , Master Washington , Jr . San Francisco , California , US Southampton San Francisco , California , US 5 Douglas , Mr. Walter Donald 50 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US 62 Douglas , Mrs. Mahala ( née Dutton ) 48 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US and maid , Miss Berthe Leroy 27 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US Douglas , Mrs. Mary Hélène ( née Baxter ) 27 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada 6 Duff Gordon , Sir Cosmo Edmund 49 London , England , UK Cherbourg New York City Duff Gordon , Lucy Christiana , Lady ( née Sutherland ) 48 London , England , UK Cherbourg New York City and secretary , Miss Laura Mabel Francatelli 31 London , England , UK Cherbourg New York City Dulles , Mr. William Crothers 39 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 133 Eustis , Miss Elizabeth Mussey 54 Brookline , Massachusetts , US Cherbourg Brookline , Massachusetts , US Evans , Miss Edith Corse 36 New York City Cherbourg New York City Flegenheim , Mrs. Antoinette ( née Wendt ) 48 New York City Cherbourg New York City 7 Flynn , Mr. John Irwin 36 Brooklyn , New York , US Southampton Brooklyn , New York , US 5 Foreman , Mr. Benjamin Laventall 30 New York City Southampton New York City Fortune , Mr. Mark 64 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Fortune , Mrs. Mary ( née McDougald ) 60 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Ethel Flora 28 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Alice Elizabeth 24 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Mabel Helen 23 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Mr. Charles Alexander 19 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Franklin , Mr. Thomas Parnham 37 London , England , UK Southampton New York City Frauenthal , Dr. Henry William 49 London , England , UK Southampton New York City 5 Frauenthal , Mrs. Clara ( née Heinsheimer ) 42 London , England , UK Southampton New York City 5 Frauenthal , Mr. Isaac Gerald 43 London , England , UK Southampton New York City 5 Frölicher , Mr. Maximilian Josef 60 Zürich , Switzerland Cherbourg New York City 5 Frölicher , Mrs. Margaretha Emerentia ( née Stehli ) 48 Zürich , Switzerland Cherbourg New York City 5 Frölicher - Stehli , Miss Hedwig Margaritha 22 Zürich , Switzerland Cherbourg New York City 5 Futrelle , Mr. Jacques Heath 37 Scituate , Massachusetts , US Southampton Scituate , Massachusetts , US Futrelle , Mrs. Lily May ( née Peel ) 35 Scituate , Massachusetts , US Southampton Scituate , Massachusetts , US Gee , Mr. Arthur H . 47 St. Annes - on - Sea , Lancashire , England , UK Southampton Mexico City , Mexico 275 Gibson , Mrs. Pauline Caroline ( née Boeson ) 44 New York City Cherbourg New York City 7 Gibson , Miss Dorothy Winifred 22 New York City Cherbourg New York City 7 Goldenberg , Mr. Samuel L . 47 Paris , France Cherbourg New York City 5 Goldenberg , Mrs. Nella ( née Wiggins ) 40 Paris , France Cherbourg New York City 5 Goldschmidt , Mr. George B . 71 New York City Cherbourg New York City Gracie IV , Colonel Archibald 53 Washington , D.C. , US Southampton Washington , D.C. , US Graham , Mr. George Edward 38 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 147 Graham , Mrs. Edith Ware ( née Junkins ) 59 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US Graham , Miss Margaret Edith 19 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US and governess , Miss Elizabeth Weed Shutes 40 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US Greenfield , Mrs. Blanche ( née Strouse ) 45 New York City Cherbourg New York City 7 Greenfield , Mr. William Bertram 23 New York City Cherbourg New York City 7 Guggenheim , Mr. Benjamin 46 Paris , France Cherbourg New York City and valet , Mr. Victor Giglio 24 Paris , France Cherbourg New York City and chauffeur , Mr. René Pernot 39 Paris , France Cherbourg New York City Harder , Mr. George Achilles 25 New York City Cherbourg New York City 5 Harder , Mrs. Dorothy ( née Annan ) 21 New York City Cherbourg New York City 5 Harper , Mr. Henry Sleeper 48 New York City Cherbourg New York City and dragoman , Mr. Hammad Hassab 27 Cairo , Egypt Cherbourg New York City Harper , Mrs. Myna ( née Haxtun ) 49 New York City Cherbourg New York City Harris , Mr. Henry Birkhardt 45 New York City Southampton New York City Harris , Mrs. Irene ( née Wallach ) 35 New York City Southampton New York City Hawksford , Mr. Walter James 45 Kingston , Surrey , England , UK Southampton New York City Hays , Mr. Charles Melville 55 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 307 and clerk , Mr. Vivian Ponsonby Payne 22 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Hays , Mrs. Clara Jennings ( née Grigg ) 52 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada and maid , Miss Mary Anne Perreault 33 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Hays , Miss Margaret Bechstein 24 New York City Cherbourg New York City 7 Head , Mr. Christopher 42 Chelsea , London , England , UK Southampton New York City Hilliard , Mr. Herbert Henry 44 Brighton , Massachusetts , US Southampton Brighton , Massachusetts , US Hipkins , Mr. William Edward 55 Birmingham , West Midlands , England , UK Southampton New York City Hippach , Mrs. Ida Sophia ( née Fischer ) 44 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Hippach , Miss Jean Gertrude 17 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Hogeboom , Mrs. Anna Louisa ( née Andrews ) 51 Hudson , New York , US Cherbourg Hudson , New York , US 10 Holverson , Mr. Alexander Oskar 42 New York City Southampton New York City 38 Holverson , Mrs. Mary Aline ( née Towner ) 35 New York City Southampton New York City 8 Homer , Mr. Harry ( alias E. Haven ) 40 Indianapolis , Indiana , US Southampton Indianapolis , Indiana , US 15 Hoyt , Mr. Frederick Maxfield 35 New York City Southampton Stamford , Connecticut , US Hoyt , Mrs. Jane Anne ( née Forby ) 31 New York City Southampton Stamford , Connecticut , US Hoyt , Mr. William Fisher 42 New York City Cherbourg New York City 14 Isham , Miss Ann Elizabeth 50 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Ismay , Mr. Joseph Bruce 49 Liverpool , Merseyside , England , UK Southampton New York City and valet , Mr. John Richard Fry 39 Liverpool , Merseyside , England , UK Southampton New York City and secretary , Mr. William Henry Harrison 45 Wallasey , Merseyside , England , UK Southampton New York City 110 Johnson , Mrs. Madeline Eliza ( née Austin ) 40 London , England , UK Southampton New York , New York , US 6 Johnson ( later changed to Langston ) , Mr. Samuel James 10 London , England , UK Southampton New York , New York , US Jones , Mr. Charles Cresson 46 Bennington , Vermont , US Southampton Bennington , Vermont , US 80 Julian , Mr. Henry Forbes 50 Torquay , Devon , England , UK Southampton San Francisco , US Kent , Mr. Edward Austin 58 Buffalo , New York , US Cherbourg Buffalo , New York , US 258 Kenyon , Mr. Frederick R . 41 Pittsburgh , Pennsylvania , US Southampton Pittsburgh , Pennsylvania , US Kenyon , Mrs. Marion ( née Stauffer ) 31 Pittsburgh , Pennsylvania , US Southampton Pittsburgh , Pennsylvania , US 8 Kimball , Mr. Edwin Nelson Jr . 42 Boston , Massachusetts , US Southampton Boston , Massachusetts , US 5 Kimball , Mrs. Gertrude ( née Parsons ) 45 Boston , Massachusetts , US Southampton Boston , Massachusetts , US 5 Klaber , Mr. Herman 45 Portland , Oregon , US Southampton Portland , Oregon , US Langston , Ms. Maxine Jane 25 Duncansville , Pennsylvania , US Southampton Duncansivlle , Pennsylvania , US Lambert - Williams , Mr. Fletcher Fellows 43 London , England , UK Southampton Newark , New Jersey , US Leader , Dr. Alice ( née Farnham ) 49 New York City Southampton New York City 8 Leslie , Lucy Noël Martha , Countess of Rothes ( née Dyer - Edwardes ) 27 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 8 and maid , Miss Roberta Elizabeth Mary `` Cissy '' Maioni 20 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 8 Lewy , Mr. Ervin G . 31 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Lindeberg - Lind , Mr. Erik Gustav ( alias Edward Lingrey ) 42 Jordanstorp , Södermanland , Sweden Southampton New York City Lindström , Mrs. Sigrid ( née Posse ) 55 Stockholm , Sweden Cherbourg New York City 6 Lines , Mrs. Elizabeth Lindsey ( née James ) 50 Paris , France Cherbourg Hanover , New Hampshire , US 9 Lines , Miss Mary Conover 16 Paris , France Cherbourg Hanover , New Hampshire , US 9 Long , Mr. Milton Clyde 29 Springfield , Massachusetts , US Southampton Springfield , Massachusetts , US 126 Longley , Miss Gretchen Fiske 21 Hudson , New York , US Cherbourg Hudson , New York , US 10 Loring , Mr. Joseph Holland 30 London , England , UK Southampton New York City Madill , Miss Georgette Alexandra 16 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Maguire , Mr. John Edward 30 Brockton , Massachusetts , US Southampton Brockton , Massachusetts , US Maréchal , Mr. Pierre , Sr . 28 Paris , France Cherbourg New York City 7 Marvin , Mr. Daniel Warner 18 New York City Southampton New York City Marvin , Mrs. Mary Graham Carmichael ( née Farquarson ) 18 New York City Southampton New York City 10 Mayné , Miss Bertha Antonine 24 Brussels , Belgium Cherbourg Montreal , Quebec , Canada 6 McCaffry , Mr. Thomas Francis 46 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 292 McCarthy , Mr. Timothy J . 54 Dorchester , Massachusetts , US Southampton Dorchester , Massachusetts , US 175 McGough , Mr. James Robert 35 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US 7 Meyer , Mr. Edgar Joseph 28 New York City Cherbourg New York City Meyer , Mrs. Leila ( née Saks ) 25 New York City Cherbourg New York City 6 Millet , Mr. Francis Davis 65 East Bridgewater , Massachusetts , US Cherbourg East Bridgewater , Massachusetts , US 249 Minahan , Dr. William Edward 44 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 230 Minahan , Mrs. Lillian E. ( née Thorpe ) 37 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 14 Minahan , Miss Daisy E . 33 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 14 Mock , Mr. Philipp Edmund 30 New York City Cherbourg New York City 11 Molson , Mr. Harry Markland 55 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Moore , Mr. Clarence Bloomfield 47 Washington , D.C. , US Southampton Washington , D.C. , US and valet , Mr. Charles Henry Harrington 37 Washington , D.C. , US Southampton Washington , D.C. , US Natsch , Mr. Charles 36 Brooklyn , New York , US Cherbourg Brooklyn , New York , US Newell , Mr. Arthur Webster 58 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 122 Newell , Miss Madeleine 31 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 6 Newell , Miss Marjorie Anne 23 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 6 Newsom , Miss Helen Monypeny 19 New York City Southampton New York City 5 Nicholson , Mr. Arthur Ernest 64 Shanklin , Isle of Wight , England , UK Southampton New York City 263 Nourney , Mr. Alfred 20 Cologne , German Empire Cherbourg New York City 7 Omont , Mr. Alfred Fernand 29 Le Havre , France Cherbourg New York City 7 Ostby , Mr. Engelhart Cornelius 64 Providence , Rhode Island , US Southampton Providence , Rhode Island , US 234 Ostby , Miss Helene Ragnhild 22 Providence , Rhode Island , US Southampton Providence , Rhode Island , US 5 Ovies y Rodriguez , Mr. Servando José Florentino 36 Havana , Cuba Cherbourg Havana , Cuba 189 Parr , Mr. William Henry Marsh 29 Belfast , Ireland , UK Belfast New York City Partner , Mr. Austin 40 Tolworth , London , England , UK Southampton Toronto , Ontario , Canada 166 Pears , Mr. Thomas Clinton 29 Isleworth , London , England , UK Southampton New York City Pears , Mrs. Edith ( née Wearne ) 22 Isleworth , London , England , UK Southampton New York City 8 Peñasco y Castellana , Mr. Victor 24 Madrid , Spain Cherbourg New York City Peñasco y Castellana , Mrs. Maria Josefa ( née Perez de Soto y Vallejo ) 22 Madrid , Spain Cherbourg New York City 8 and maid , Doña Fermina Oliva y Ocana 39 Madrid , Spain Cherbourg New York City 8 Peuchen , Major Arthur Godfrey 52 Toronto , Ontario , Canada Southampton Toronto , Ontario , Canada 6 Porter , Mr. Walter Chamberlain 46 Worcester , Massachusetts , US Southampton Worcester , Massachusetts , US 207 Potter , Mrs. Lily Alexenia ( née Wilson ) 56 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 7 Reuchlin , The Honourable Mr. Johan George 38 Rotterdam , The Netherlands Southampton New York City Rheims , Mr. George Alexander Lucien 36 Paris , France Southampton New York City Robert , Mrs. Elisabeth Walton ( née McMillan ) 43 St. Louis , Missouri , US Southampton St. Louis , Missouri , US and maid , Miss Emilie Kreuchen 29 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Roebling , Mr. Washington Augustus II 31 Trenton , New Jersey , US Southampton Trenton , New Jersey , US Romaine , Mr. Charles Hallance ( alias C. Rolmane ) 45 New York City Southampton New York City 9 Rood , Mr. Hugh Roscoe 38 Seattle , Washington , US Southampton Seattle , Washington , US Rosenbaum , Miss Edith Louise 34 Paris , France Cherbourg Cincinnati , Ohio , US 11 Rosenshine , Mr. George ( alias George Thorne ) 46 New York City Cherbourg New York City 16 Ross , Mr. John Hugo 36 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Rothschild , Mr. Martin 46 New York City Cherbourg New York City Rothschild , Mrs. Elizabeth Jane Anne ( née Barrett ) 54 New York City Cherbourg New York City 6 Rowe , Mr. Alfred G . 59 Liverpool , Merseyside , England , UK Southampton New York City 109 Ryerson , Mr. Arthur Larned 61 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Mrs. Emily Maria ( née Borie ) 48 Cooperstown , New York , US Cherbourg Cooperstown , New York , US and maid , Miss Victorine Chaudanson 36 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Miss Susan Parker `` Suzette '' 21 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Miss Emily Borie 18 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Master John Borie `` Jack '' 13 Cooperstown , New York , US Cherbourg Cooperstown , New York , US and governess , Miss Grace Scott Bowen 45 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Saalfeld , Mr. Adolphe 47 Manchester , England , UK Southampton New York City Salomon , Mr. Abraham Lincoln 43 New York City Cherbourg New York City Schabert , Mrs. Emma ( née Mock ) 35 Hamburg , German Empire Southampton New York City 11 Seward , Mr. Frederic Kimber 34 New York City Southampton New York City 7 Silverthorne , Mr. Spencer Victor 35 St. Louis , Missouri , US Southampton St. Louis , Missouri , US 5 Silvey , Mr. William Baird 50 Duluth , Minnesota , US Cherbourg Duluth , Minnesota , US Silvey , Mrs. Alice ( née Munger ) 39 Duluth , Minnesota , US Cherbourg Duluth , Minnesota , US 11 Simonius - Blumer , Colonel Alfons 56 Basel , Switzerland Southampton New York City Sloper , Mr. William Thompson 28 New Britain , Connecticut , US Southampton New Britain , Connecticut , US 7 Smart , Mr. John Montgomery 56 Kildale , North Yorkshire , England , UK Southampton New York City Smith , Mr. James Clinch 56 Paris , France Cherbourg Long Island , New York , US Smith , Mr. Lucien Philip 24 Huntington , West Virginia , US Cherbourg Huntington , West Virginia , US Smith , Mrs. Mary Eloise ( née Hughes ) 18 Huntington , West Virginia , US Cherbourg Huntington , West Virginia , US 6 Smith , Mr. Richard William 57 Streatham , London , England , UK Southampton New York City Snyder , Mr. John Pillsbury 24 Minneapolis , Minnesota , US Southampton Minneapolis , Minnesota , US 7 Snyder , Mrs. Nellie ( née Stevenson ) 23 Minneapolis , Minnesota , US Southampton Minneapolis , Minnesota , US 7 Spedden , Mr. Frederic Oakley 45 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spedden , Mrs. Margaretta Corning ( née Stone ) 39 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US and maid , Miss Helen Alice Wilson 31 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spedden , Master Robert Douglas 6 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US and nurse , Miss Elizabeth Margaret Burns 41 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spencer , Mr. William Augustus 57 New York City Cherbourg New York City Spencer , Mrs. Marie Eugénie ( née Demougeot ) 45 New York City Cherbourg New York City 6 and maid , Miss Eugenie Elise Lurette 59 New York City Cherbourg New York City 6 Stähelin - Maeglin , Dr. Max 32 Basel , Switzerland Southampton New York City Stead , Mr. William Thomas 62 London , England , UK Southampton New York City Stengel , Mr. Charles Emil Henry 54 Newark , New Jersey , US Cherbourg Newark , New Jersey , US Stengel , Mrs. Annie May ( née Morris ) 44 Newark , New Jersey , US Cherbourg Newark , New Jersey , US 5 Stephenson , Mrs. Martha ( née Eustis ) 52 Paris , France Cherbourg Haverford , Pennsylvania , US Stewart , Mr. Albert A . 54 Cincinnati , Ohio , US Cherbourg Gallipolis , Ohio , US Stone , Mrs. Martha Evelyn ( née Stevens ) 62 New York City Southampton New York City 6 and maid , Miss Amelie `` Amelia '' Icard 38 New York City Southampton New York City 6 Straus , Mr. Isidor 67 New York City Southampton New York City 96 and valet , Mr. John Farthing 57 New York City Southampton New York City Straus , Mrs. Rosalie Ida ( née Blun ) 63 New York City Southampton New York City and maid , Miss Ellen Bird 31 New York City Southampton New York City 8 Sutton , Mr. Frederick 61 Haddonfield , New Jersey , US Southampton Haddonfield , New Jersey , US 46 Swift , Mrs. Margaret Welles ( née Barron ) 46 New York City Southampton New York City 8 Taussig , Mr. Emil 52 New York City Southampton New York City Taussig , Mrs. Tillie ( née Mandelbaum ) 39 New York City Southampton New York City 8 Taussig , Miss Ruth 18 New York City Southampton New York City 8 Taylor , Mr. Elmer Zebley 48 London , England , UK Southampton East Orange , New Jersey , US 5 Taylor , Mrs. Juliet Cummins ( née Wright ) 48 London , England , UK Southampton East Orange , New Jersey , US 5 Thayer , Mr. John Borland II 49 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thayer , Mrs. Marian Longsteth ( née Morris ) 39 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US and maid , Miss Margaret Fleming 42 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thayer , Mr. John Borland `` Jack '' III 17 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thorne , Mrs. Gertrude Maybelle ( née McMinn ) 37 New York City , New York , US Cherbourg New York City , New York , US Tucker , Mr. Gilbert Milligan Jr . 31 Albany , New York , US Cherbourg Albany , New York , US 7 Uruchurtu , Don Manuel E . 40 Mexico City , Mexico Cherbourg Mexico City , Mexico Van der Hoef , Mr. Wyckoff 61 Brooklyn , New York , US Belfast Brooklyn , New York , US 245 Walker , Mr. William Anderson 48 East Orange , New Jersey , US Southampton East Orange , New Jersey , US Warren , Mr. Frank Manley 63 Portland , Oregon , US Cherbourg Portland , Oregon , US Warren , Mrs. Anna Sophia ( née Atkinson ) 60 Portland , Oregon , US Cherbourg Portland , Oregon , US 5 Weir , Colonel John 59 New York City Southampton New York City White , Mr. Percival Wayland 54 Brunswick , Maine , US Southampton Brunswick , Maine , US White , Mr. Richard Frasar 21 Brunswick , Maine , US Southampton Brunswick , Maine , US 169 White , Mrs. Ella ( née Holmes ) 55 New York City Cherbourg New York City 8 and maid , Miss Amelia Mayo `` Nellie '' Bessette 39 New York City Cherbourg New York City 8 and manservant , Mr. Sante Righini 22 New York City Cherbourg New York City 232 Wick , Colonel George Dennick 58 Youngstown , Ohio , US Southampton Youngstown , Ohio , US Wick , Mrs. Mary ( née Hitchcock ) 45 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Wick , Miss Mary Natalie 31 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Widener , Mr. George Dunton 50 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US and valet , Mr. Edwin Herbert Keeping 33 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US 45 Widener , Mrs. Eleanor ( née Elkins ) 50 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US and maid , Miss Amalie Henriette `` Emily '' Gieger 35 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Widener , Mr. Harry Elkins 27 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Willard , Miss Constance 21 Duluth , Minnesota , US Southampton Duluth , Minnesota , US 8 Williams , Mr. Charles Duane 51 Geneva , Switzerland Cherbourg Radnor , Pennsylvania , US Williams , Mr. Richard Norris II 21 Geneva , Switzerland Cherbourg Radnor , Pennsylvania , US Woolner , Mr. Hugh 45 London , England , UK Southampton New York City Wright , Mr. George 62 Halifax , Nova Scotia , Canada Southampton Halifax , Nova Scotia , Canada Young , Miss Marie Grice 36 New York City Cherbourg New York City 8
  Fact 0.0: Allen, Miss Elizabeth Walton was 29 years old.
  Fact 0.1: Allen, Miss Elizabeth Walton is from St. Louis, Missouri, US.
  Fact 0.2: Allen, Miss Elizabeth Walton boarded at Southampton.
  Fact 0.3: Allen, Miss Elizabeth Walton's destination is St. Louis, Missouri, US.
  Fact 0.4: Allison, Mr. Hudson Joshua Creighton was 30 years old.
  Fact 0.5: Allison, Mr. Hudson Joshua Creighton is from Montreal, Quebec, Canada.
  Fact 0.6: Allison, Mr. Hudson Joshua Creighton boarded at Southampton.
  Fact 0.7: Allison, Mr. Hudson Joshua Creighton's destination is Montreal, Quebec, Canada.
  Fact 0.8: Allison, Mr. Hudson Joshua Creighton has 135 in the lifeboat or body column.
  Fact 0.9: Mr. George Swane was 19 years old.
  Fact 0.10: Mr. George Swane is listed as the chauffeur.
  Fact 0.11: Mr. George Swane is from Montreal, Quebec, Canada.
  Fact 0.12: Mr. George Swane boarded at Southampton.
  Fact 0.13: Mr. George Swane's destination is Montreal, Quebec, Canada.
  Fact 0.14: Mr. George Swane has 294 in the lifeboat or body column.
  Fact 0.15: Miss Amelia Mary "Mildred" Brown was 18 years old.
  Fact 0.16: Miss Amelia Mary "Mildred" Brown is listed as the cook.
  Fact 0.17: Miss Amelia Mary "Mildred" Brown is from London, England, UK.
  Fact 0.18: Miss Amelia Mary "Mildred" Brown boarded at Southampton.
  Fact 0.19: Miss Amelia Mary "Mildred" Brown's destination is Montreal, Quebec, Canada.
  Fact 0.20: Miss Amelia Mary "Mildred" Brown has 11 in the lifeboat or body column.
  Fact 0.21: Allison, Mrs. Bessie Waldo (née Daniels) was 25 years old.
  Fact 0.22: Allison, Mrs. Bessie Waldo (née Daniels) is from Montreal, Quebec, Canada.
  Fact 0.23: Allison, Mrs. Bessie Waldo (née Daniels) boarded at Southampton.
  Fact 0.24: Allison, Mrs. Bessie Waldo (née Daniels)'s destination is Montreal, Quebec, Canada.
  Fact 0.25: Miss Sarah Daniels was 33 years old.
  Fact 0.26: Miss Sarah Daniels is listed as the maid.
  Fact 0.27: Miss Sarah Daniels is from Montreal, Quebec, Canada.
  Fact 0.28: Miss Sarah Daniels boarded at Southampton.
  Fact 0.29: Miss Sarah Daniels's destination is Montreal, Quebec, Canada.
  Fact 0.30: Miss Sarah Daniels has 8 in the lifeboat or body column.
  Fact 0.31: Allison, Miss Helen Loraine is from Montreal, Quebec, Canada.
  Fact 0.32: Allison, Miss Helen Loraine boarded at Southampton.
  Fact 0.33: Allison, Miss Helen Loraine's destination is Montreal, Quebec, Canada.
  Fact 0.34: Allison, Master Hudson Trevor is listed as age 0.11 and as 11 months.
  Fact 0.35: Allison, Master Hudson Trevor is from Montreal, Quebec, Canada.
  Fact 0.36: Allison, Master Hudson Trevor boarded at Southampton.
  Fact 0.37: Allison, Master Hudson Trevor's destination is Montreal, Quebec, Canada.
  Fact 0.38: Allison, Master Hudson Trevor has 11 in the lifeboat or body column.
  Fact 0.39: Miss Alice Catherine Cleaver was 22 years old.
  Fact 0.40: Miss Alice Catherine Cleaver is listed as the nurse.
  Fact 0.41: Miss Alice Catherine Cleaver is from London, England, UK.
  Fact 0.42: Miss Alice Catherine Cleaver boarded at Southampton.
  Fact 0.43: Miss Alice Catherine Cleaver's destination is Montreal, Quebec, Canada.
  Fact 0.44: Miss Alice Catherine Cleaver has 11 in the lifeboat or body column.
  Fact 0.45: Anderson, Mr. Harry was 47 years old.
  Fact 0.46: Anderson, Mr. Harry is from New York City, New York, US.
  Fact 0.47: Anderson, Mr. Harry boarded at Southampton.
  Fact 0.48: Anderson, Mr. Harry's destination is New York City, New York, US.
  Fact 0.49: Andrews, Miss Kornelia Theodosia was 62 years old.
  Fact 0.50: Andrews, Miss Kornelia Theodosia is from Hudson, New York, US.
  Fact 0.51: Andrews, Miss Kornelia Theodosia boarded at Cherbourg.
  Fact 0.52: Andrews, Miss Kornelia Theodosia's destination is Hudson, New York, US.
  Fact 0.53: Andrews, Miss Kornelia Theodosia has 10 in the lifeboat or body column.
  Fact 0.54: Andrews, Mr. Thomas was 39 years old.
  Fact 0.55: Andrews, Mr. Thomas is from Belfast, Ireland, UK.
  Fact 0.56: Andrews, Mr. Thomas boarded at Belfast.
  Fact 0.57: Andrews, Mr. Thomas's destination is New York City.
  Fact 0.58: Appleton, Mrs. Charlotte Lane (née Lamson) was 53 years old.
  Fact 0.59: Appleton, Mrs. Charlotte Lane (née Lamson) is from New York City.
  Fact 0.60: Appleton, Mrs. Charlotte Lane (née Lamson) boarded at Southampton.
  Fact 0.61: Appleton, Mrs. Charlotte Lane (née Lamson)'s destination is New York City.
  Fact 0.62: Artagaveytia, Mr. Ramon was 71 years old.
  Fact 0.63: Artagaveytia, Mr. Ramon is from Buenos Aires, Argentina.
  Fact 0.64: Artagaveytia, Mr. Ramon boarded at Cherbourg.
  Fact 0.65: Artagaveytia, Mr. Ramon's destination is New York City.
  Fact 0.66: Artagaveytia, Mr. Ramon has 22 in the lifeboat or body column.
  Fact 0.67: Astor, Colonel John Jacob IV was 47 years old.
  Fact 0.68: Astor, Colonel John Jacob IV is from New York City.
  Fact 0.69: Astor, Colonel John Jacob IV boarded at Cherbourg.
  Fact 0.70: Astor, Colonel John Jacob IV's destination is New York City.
  Fact 0.71: Astor, Colonel John Jacob IV has 124 in the lifeboat or body column.
  Fact 0.72: Astor, Mrs. Madeleine Talmage (née Force) was 18 years old.
  Fact 0.73: Astor, Mrs. Madeleine Talmage (née Force) is from New York City.
  Fact 0.74: Astor, Mrs. Madeleine Talmage (née Force) boarded at Cherbourg.
  Fact 0.75: Astor, Mrs. Madeleine Talmage (née Force)'s destination is New York City.
  Fact 0.76: Miss Rosalie Bidois was 46 years old.
  Fact 0.77: Miss Rosalie Bidois is listed as the maid.
  Fact 0.78: Miss Rosalie Bidois is from New York City.
  Fact 0.79: Miss Rosalie Bidois boarded at Cherbourg.
  Fact 0.80: Miss Rosalie Bidois's destination is New York City.
  Fact 0.81: Miss Caroline Louise Endres was 39 years old.
  Fact 0.82: Miss Caroline Louise Endres is listed as the nurse.
  Fact 0.83: Miss Caroline Louise Endres is from New York City.
  Fact 0.84: Miss Caroline Louise Endres boarded at Cherbourg.
  Fact 0.85: Miss Caroline Louise Endres's destination is New York City.
  Fact 0.86: Aubart, Mrs. Léontine Pauline was 24 years old.
  Fact 0.87: Aubart, Mrs. Léontine Pauline is from Paris, France.
  Fact 0.88: Aubart, Mrs. Léontine Pauline boarded at Cherbourg.
  Fact 0.89: Aubart, Mrs. Léontine Pauline's destination is New York City.
  Fact 0.90: Aubart, Mrs. Léontine Pauline has 9 in the lifeboat or body column.
  Fact 0.91: Miss Emma Sägesser was 24 years old.
  Fact 0.92: Miss Emma Sägesser is listed as the maid.
  Fact 0.93: Miss Emma Sägesser is from Paris, France.
  Fact 0.94: Miss Emma Sägesser boarded at Cherbourg.
  Fact 0.95: Miss Emma Sägesser's destination is New York City.
  Fact 0.96: Miss Emma Sägesser has 9 in the lifeboat or body column.
  Fact 0.97: Barkworth, Mr. Algernon Henry Wilson was 47 years old.
  Fact 0.98: Barkworth, Mr. Algernon Henry Wilson is from Hessle, East Yorkshire, England, UK.
  Fact 0.99: Barkworth, Mr. Algernon Henry Wilson boarded at Southampton.
  Fact 0.100: Barkworth, Mr. Algernon Henry Wilson's destination is New York City.
  Fact 0.101: Baumann, Mr. John D. was 60 years old.
  Fact 0.102: Baumann, Mr. John D. is from New York City.
  Fact 0.103: Baumann, Mr. John D. boarded at Cherbourg.
  Fact 0.104: Baumann, Mr. John D.'s destination is New York City.
  Fact 0.105: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) was 50 years old.
  Fact 0.106: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) is from Montreal, Quebec, Canada.
  Fact 0.107: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) boarded at Cherbourg.
  Fact 0.108: Baxter, Mrs. Hélène (née de Lanaudière-Chaput)'s destination is Montreal, Quebec, Canada.
  Fact 0.109: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) has 6 in the lifeboat or body column.
  Fact 0.110: Baxter, Mr. Quigg Edmond was 24 years old.
  Fact 0.111: Baxter, Mr. Quigg Edmond is from Montreal, Quebec, Canada.
  Fact 0.112: Baxter, Mr. Quigg Edmond boarded at Cherbourg.
  Fact 0.113: Baxter, Mr. Quigg Edmond's destination is Montreal, Quebec, Canada.
  Fact 0.114: Beattie, Mr. Thomson was 36 years old.
  Fact 0.115: Beattie, Mr. Thomson is from Fergus, Ontario, Canada.
  Fact 0.116: Beattie, Mr. Thomson boarded at Southampton.
  Fact 0.117: Beattie, Mr. Thomson's destination is Fergus, Ontario, Canada.
  Fact 0.118: Beattie, Mr. Thomson has 331 in the lifeboat or body column.
  Fact 0.119: Beckwith, Mr. Richard Leonard was 37 years old.
  Fact 0.120: Beckwith, Mr. Richard Leonard is from New York City.
  Fact 0.121: Beckwith, Mr. Richard Leonard boarded at Southampton.
  Fact 0.122: Beckwith, Mr. Richard Leonard's destination is New York City.
  Fact 0.123: Beckwith, Mr. Richard Leonard has 5 in the lifeboat or body column.
  Fact 0.124: Beckwith, Mrs. Sallie (née Monypeny) was 46 years old.
  Fact 0.125: Beckwith, Mrs. Sallie (née Monypeny) is from New York City.
  Fact 0.126: Beckwith, Mrs. Sallie (née Monypeny) boarded at Southampton.
  Fact 0.127: Beckwith, Mrs. Sallie (née Monypeny)'s destination is New York City.
  Fact 0.128: Beckwith, Mrs. Sallie (née Monypeny) has 5 in the lifeboat or body column.
  Fact 0.129: Behr, Mr. Karl Howell was 26 years old.
  Fact 0.130: Behr, Mr. Karl Howell is from New York City.
  Fact 0.131: Behr, Mr. Karl Howell boarded at Cherbourg.
  Fact 0.132: Behr, Mr. Karl Howell's destination is New York City.
  Fact 0.133: Behr, Mr. Karl Howell has 5 in the lifeboat or body column.
  Fact 0.134: Birnbaum, Mr. Jakob was 24 years old.
  Fact 0.135: Birnbaum, Mr. Jakob is from Antwerp, Belgium.
  Fact 0.136: Birnbaum, Mr. Jakob boarded at Cherbourg.
  Fact 0.137: Birnbaum, Mr. Jakob's destination is San Francisco, California, US.
  Fact 0.138: Birnbaum, Mr. Jakob has 148 in the lifeboat or body column.
  Fact 0.139: Bishop, Mr. Dickinson H. "Dick" was 25 years old.
  Fact 0.140: Bishop, Mr. Dickinson H. "Dick" is from Dowagiac, Michigan, US.
  Fact 0.141: Bishop, Mr. Dickinson H. "Dick" boarded at Cherbourg.
  Fact 0.142: Bishop, Mr. Dickinson H. "Dick"'s destination is Dowagiac, Michigan, US.
  Fact 0.143: Bishop, Mr. Dickinson H. "Dick" has 7 in the lifeboat or body column.
  Fact 0.144: Bishop, Mrs. Helen (née Walton) was 19 years old.
  Fact 0.145: Bishop, Mrs. Helen (née Walton) is from Dowagiac, Michigan, US.
  Fact 0.146: Bishop, Mrs. Helen (née Walton) boarded at Cherbourg.
  Fact 0.147: Bishop, Mrs. Helen (née Walton)'s destination is Dowagiac, Michigan, US.
  Fact 0.148: Bishop, Mrs. Helen (née Walton) has 7 in the lifeboat or body column.
  Fact 0.149: Björnström-Steffanson, Mr. Mauritz Håkan was 28 years old.
  Fact 0.150: Björnström-Steffanson, Mr. Mauritz Håkan is from Stockholm, Sweden.
  Fact 0.151: Björnström-Steffanson, Mr. Mauritz Håkan boarded at Southampton.
  Fact 0.152: Björnström-Steffanson, Mr. Mauritz Håkan's destination is Washington, D.C., US.
  Fact 0.153: Blackwell, Mr. Stephen Weart was 45 years old.
  Fact 0.154: Blackwell, Mr. Stephen Weart is from Trenton, New Jersey, US.
  Fact 0.155: Blackwell, Mr. Stephen Weart boarded at Southampton.
  Fact 0.156: Blackwell, Mr. Stephen Weart's destination is Trenton, New Jersey, US.
  Fact 0.157: Blank, Mr. Henry was 39 years old.
  Fact 0.158: Blank, Mr. Henry is from Glen Ridge, New Jersey, US.
  Fact 0.159: Blank, Mr. Henry boarded at Cherbourg.
  Fact 0.160: Blank, Mr. Henry's destination is Glen Ridge, New Jersey, US.
  Fact 0.161: Blank, Mr. Henry has 7 in the lifeboat or body column.
  Fact 0.162: Bonnell, Miss Elizabeth was 61 years old.
  Fact 0.163: Bonnell, Miss Elizabeth is from Youngstown, Ohio, US.
  Fact 0.164: Bonnell, Miss Elizabeth boarded at Southampton.
  Fact 0.165: Bonnell, Miss Elizabeth's destination is Youngstown, Ohio, US.
  Fact 0.166: Bonnell, Miss Elizabeth has 8 in the lifeboat or body column.
  Fact 0.167: Bonnell, Miss Caroline was 30 years old.
  Fact 0.168: Bonnell, Miss Caroline is from Youngstown, Ohio, US.
  Fact 0.169: Bonnell, Miss Caroline boarded at Southampton.
  Fact 0.170: Bonnell, Miss Caroline's destination is Youngstown, Ohio, US.
  Fact 0.171: Bonnell, Miss Caroline has 8 in the lifeboat or body column.
  Fact 0.172: Borebank, Mr. John James was 42 years old.
  Fact 0.173: Borebank, Mr. John James is from London, England, UK.
  Fact 0.174: Borebank, Mr. John James boarded at Southampton.
  Fact 0.175: Borebank, Mr. John James's destination is Toronto, Ontario, Canada.
  Fact 0.176: Bowerman, Miss Elsie Edith was 22 years old.
  Fact 0.177: Bowerman, Miss Elsie Edith is from St. Leonards-on-Sea, East Sussex, England, UK.
  Fact 0.178: Bowerman, Miss Elsie Edith boarded at Southampton.
  Fact 0.179: Bowerman, Miss Elsie Edith's destination is New York City.
  Fact 0.180: Bowerman, Miss Elsie Edith has 6 in the lifeboat or body column.
  Fact 0.181: Brady, Mr. John Bertram was 41 years old.
  Fact 0.182: Brady, Mr. John Bertram is from Pomeroy, Washington, US.
  Fact 0.183: Brady, Mr. John Bertram boarded at Southampton.
  Fact 0.184: Brady, Mr. John Bertram's destination is Pomeroy, Washington, US.
  Fact 0.185: Brandeis, Mr. Emil was 48 years old.
  Fact 0.186: Brandeis, Mr. Emil is from Omaha, Nebraska, US.
  Fact 0.187: Brandeis, Mr. Emil boarded at Cherbourg.
  Fact 0.188: Brandeis, Mr. Emil's destination is Omaha, Nebraska, US.
  Fact 0.189: Brandeis, Mr. Emil has 208 in the lifeboat or body column.
  Fact 0.190: Brereton, Mr. George Andrew (alias George A. Brayton) was 37 years old.
  Fact 0.191: Brereton, Mr. George Andrew (alias George A. Brayton) is from Los Angeles, US.
  Fact 0.192: Brereton, Mr. George Andrew (alias George A. Brayton) boarded at Southampton.
  Fact 0.193: Brereton, Mr. George Andrew (alias George A. Brayton)'s destination is Los Angeles, California, US.
  Fact 0.194: Brereton, Mr. George Andrew (alias George A. Brayton) has 9 in the lifeboat or body column.
  Fact 0.195: Brewe, Dr. Arthur Jackson was 45 years old.
  Fact 0.196: Brewe, Dr. Arthur Jackson is from Philadelphia, Pennsylvania, US.
  Fact 0.197: Brewe, Dr. Arthur Jackson boarded at Cherbourg.
  Fact 0.198: Brewe, Dr. Arthur Jackson's destination is Philadelphia, Pennsylvania, US.
  Fact 0.199: Brown, Mrs. Caroline Lane (née Lamson) was 59 years old.
  Fact 0.200: Brown, Mrs. Caroline Lane (née Lamson) is from Belmont, Massachusetts, US.
  Fact 0.201: Brown, Mrs. Caroline Lane (née Lamson) boarded at Southampton.
  Fact 0.202: Brown, Mrs. Caroline Lane (née Lamson)'s destination is Belmont, Massachusetts, US.
  Fact 0.203: Brown, Mrs. Margaret (née Tobin) was 44 years old.
  Fact 0.204: Brown, Mrs. Margaret (née Tobin) is from Denver, Colorado, US.
  Fact 0.205: Brown, Mrs. Margaret (née Tobin) boarded at Cherbourg.
  Fact 0.206: Brown, Mrs. Margaret (née Tobin)'s destination is Denver, Colorado, US.
  Fact 0.207: Brown, Mrs. Margaret (née Tobin) has 6 in the lifeboat or body column.
  Fact 0.208: Bucknell, Mrs. Emma Eliza (née Ward) was 59 years old.
  Fact 0.209: Bucknell, Mrs. Emma Eliza (née Ward) is from Philadelphia, Pennsylvania, US.
  Fact 0.210: Bucknell, Mrs. Emma Eliza (née Ward) boarded at Cherbourg.
  Fact 0.211: Bucknell, Mrs. Emma Eliza (née Ward)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.212: Bucknell, Mrs. Emma Eliza (née Ward) has 8 in the lifeboat or body column.
  Fact 0.213: Miss Albina Bazzani was 36 years old.
  Fact 0.214: Miss Albina Bazzani is listed as the maid.
  Fact 0.215: Miss Albina Bazzani is from Philadelphia, Pennsylvania, US.
  Fact 0.216: Miss Albina Bazzani boarded at Cherbourg.
  Fact 0.217: Miss Albina Bazzani's destination is Philadelphia, Pennsylvania, US.
  Fact 0.218: Miss Albina Bazzani has 8 in the lifeboat or body column.
  Fact 0.219: Butt, Major Archibald Willingham was 46 years old.
  Fact 0.220: Butt, Major Archibald Willingham is from Washington, D.C., US.
  Fact 0.221: Butt, Major Archibald Willingham boarded at Southampton.
  Fact 0.222: Butt, Major Archibald Willingham's destination is Washington, D.C., US.
  Fact 0.223: Calderhead, Mr. Edward Pennington was 42 years old.
  Fact 0.224: Calderhead, Mr. Edward Pennington is from New York City.
  Fact 0.225: Calderhead, Mr. Edward Pennington boarded at Southampton.
  Fact 0.226: Calderhead, Mr. Edward Pennington's destination is New York City.
  Fact 0.227: Calderhead, Mr. Edward Pennington has 5 in the lifeboat or body column.
  Fact 0.228: Candee, Mrs. Helen Churchill (née Hungerford) was 53 years old.
  Fact 0.229: Candee, Mrs. Helen Churchill (née Hungerford) is from Washington, D.C., US.
  Fact 0.230: Candee, Mrs. Helen Churchill (née Hungerford) boarded at Cherbourg.
  Fact 0.231: Candee, Mrs. Helen Churchill (née Hungerford)'s destination is Washington, D.C., US.
  Fact 0.232: Candee, Mrs. Helen Churchill (née Hungerford) has 6 in the lifeboat or body column.
  Fact 0.233: Cardeza, Mrs. Charlotte Wardle (née Drake) was 58 years old.
  Fact 0.234: Cardeza, Mrs. Charlotte Wardle (née Drake) is from Germantown, Pennsylvania, US.
  Fact 0.235: Cardeza, Mrs. Charlotte Wardle (née Drake) boarded at Cherbourg.
  Fact 0.236: Cardeza, Mrs. Charlotte Wardle (née Drake)'s destination is Germantown, Pennsylvania, US.
  Fact 0.237: Miss Annie Moore Ward was 38 years old.
  Fact 0.238: Miss Annie Moore Ward is listed as the maid.
  Fact 0.239: Miss Annie Moore Ward is from Germantown, Pennsylvania, US.
  Fact 0.240: Miss Annie Moore Ward boarded at Cherbourg.
  Fact 0.241: Miss Annie Moore Ward's destination is Germantown, Pennsylvania, US.
  Fact 0.242: Cardeza, Mr. Thomas Drake Martinez was 36 years old.
  Fact 0.243: Cardeza, Mr. Thomas Drake Martinez is from Germantown, Pennsylvania, US.
  Fact 0.244: Cardeza, Mr. Thomas Drake Martinez boarded at Cherbourg.
  Fact 0.245: Cardeza, Mr. Thomas Drake Martinez's destination is Germantown, Pennsylvania, US.
  Fact 0.246: Mr. Gustave J. Lesueur was 35 years old.
  Fact 0.247: Mr. Gustave J. Lesueur is listed as the valet.
  Fact 0.248: Mr. Gustave J. Lesueur is from Germantown, Pennsylvania, US.
  Fact 0.249: Mr. Gustave J. Lesueur boarded at Cherbourg.
  Fact 0.250: Mr. Gustave J. Lesueur's destination is Germantown, Pennsylvania, US.
  Fact 0.251: Carlsson, Mr. Frans Olof was 33 years old.
  Fact 0.252: Carlsson, Mr. Frans Olof is from New York City.
  Fact 0.253: Carlsson, Mr. Frans Olof boarded at Southampton.
  Fact 0.254: Carlsson, Mr. Frans Olof's destination is New York City.
  Fact 0.255: Carrau, Mr. Francisco M. was 31 years old.
  Fact 0.256: Carrau, Mr. Francisco M. is from Montevideo, Uruguay.
  Fact 0.257: Carrau, Mr. Francisco M. boarded at Southampton.
  Fact 0.258: Carrau, Mr. Francisco M.'s destination is Montevideo, Uruguay.
  Fact 0.259: Carrau, Mr. José Pedro was 17 years old.
  Fact 0.260: Carrau, Mr. José Pedro is from Montevideo, Uruguay.
  Fact 0.261: Carrau, Mr. José Pedro boarded at Southampton.
  Fact 0.262: Carrau, Mr. José Pedro's destination is Montevideo, Uruguay.
  Fact 0.263: Carter, Mr. William Ernest was 36 years old.
  Fact 0.264: Carter, Mr. William Ernest is from Bryn Mawr, Pennsylvania, US.
  Fact 0.265: Carter, Mr. William Ernest boarded at Southampton.
  Fact 0.266: Carter, Mr. William Ernest's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.267: Mr. Alexander Cairns was 28 years old.
  Fact 0.268: Mr. Alexander Cairns is listed as the valet.
  Fact 0.269: Mr. Alexander Cairns is from Bryn Mawr, Pennsylvania, US.
  Fact 0.270: Mr. Alexander Cairns boarded at Southampton.
  Fact 0.271: Mr. Alexander Cairns's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.272: Mr. Charles Augustus Aldworth was 30 years old.
  Fact 0.273: Mr. Charles Augustus Aldworth is listed as the chauffeur.
  Fact 0.274: Mr. Charles Augustus Aldworth is from Bryn Mawr, Pennsylvania, US.
  Fact 0.275: Mr. Charles Augustus Aldworth boarded at Southampton.
  Fact 0.276: Mr. Charles Augustus Aldworth's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.277: Carter, Mrs. Lucile (née Polk) was 36 years old.
  Fact 0.278: Carter, Mrs. Lucile (née Polk) is from Bryn Mawr, Pennsylvania, US.
  Fact 0.279: Carter, Mrs. Lucile (née Polk) boarded at Southampton.
  Fact 0.280: Carter, Mrs. Lucile (née Polk)'s destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.281: Miss Auguste Serreplan was 30 years old.
  Fact 0.282: Miss Auguste Serreplan is listed as the maid.
  Fact 0.283: Miss Auguste Serreplan is from Bryn Mawr, Pennsylvania, US.
  Fact 0.284: Miss Auguste Serreplan boarded at Southampton.
  Fact 0.285: Miss Auguste Serreplan's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.286: Carter, Miss Lucile Polk was 13 years old.
  Fact 0.287: Carter, Miss Lucile Polk is from Bryn Mawr, Pennsylvania, US.
  Fact 0.288: Carter, Miss Lucile Polk boarded at Southampton.
  Fact 0.289: Carter, Miss Lucile Polk's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.290: Carter, Master William Thornton II was 11 years old.
  Fact 0.291: Carter, Master William Thornton II is from Bryn Mawr, Pennsylvania, US.
  Fact 0.292: Carter, Master William Thornton II boarded at Southampton.
  Fact 0.293: Carter, Master William Thornton II's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.294: Case, Mr. Howard Brown was 49 years old.
  Fact 0.295: Case, Mr. Howard Brown is from Ascot, Berkshire, England, UK.
  Fact 0.296: Case, Mr. Howard Brown boarded at Southampton.
  Fact 0.297: Case, Mr. Howard Brown's destination is Rochester, New York, US.
  Fact 0.298: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) was 36 years old.
  Fact 0.299: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) is from New York City.
  Fact 0.300: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) boarded at Cherbourg.
  Fact 0.301: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick)'s destination is New York City.
  Fact 0.302: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) has 5 in the lifeboat or body column.
  Fact 0.303: Cavendish, Mr. Tyrell William was 36 years old.
  Fact 0.304: Cavendish, Mr. Tyrell William is from London, England, UK.
  Fact 0.305: Cavendish, Mr. Tyrell William boarded at Southampton.
  Fact 0.306: Cavendish, Mr. Tyrell William's destination is New York City.
  Fact 0.307: Cavendish, Mr. Tyrell William has 172 in the lifeboat or body column.
  Fact 0.308: Cavendish, Mrs. Julia Florence (née Siegel) was 25 years old.
  Fact 0.309: Cavendish, Mrs. Julia Florence (née Siegel) is from London, England, UK.
  Fact 0.310: Cavendish, Mrs. Julia Florence (née Siegel) boarded at Southampton.
  Fact 0.311: Cavendish, Mrs. Julia Florence (née Siegel)'s destination is New York City.
  Fact 0.312: Cavendish, Mrs. Julia Florence (née Siegel) has 6 in the lifeboat or body column.
  Fact 0.313: Miss Ellen "Nellie" Barber was 26 years old.
  Fact 0.314: Miss Ellen "Nellie" Barber is listed as the maid.
  Fact 0.315: Miss Ellen "Nellie" Barber is from London, England, UK.
  Fact 0.316: Miss Ellen "Nellie" Barber boarded at Southampton.
  Fact 0.317: Miss Ellen "Nellie" Barber's destination is New York City.
  Fact 0.318: Miss Ellen "Nellie" Barber has 6 in the lifeboat or body column.
  Fact 0.319: Chaffee, Mr. Hubert Fuller was 46 years old.
  Fact 0.320: Chaffee, Mr. Hubert Fuller is from Amenia, North Dakota, US.
  Fact 0.321: Chaffee, Mr. Hubert Fuller boarded at Southampton.
  Fact 0.322: Chaffee, Mr. Hubert Fuller's destination is Amenia, North Dakota, US.
  Fact 0.323: Chaffee, Mrs. Carrie Constance (née Toogood) was 47 years old.
  Fact 0.324: Chaffee, Mrs. Carrie Constance (née Toogood) is from Amenia, North Dakota, US.
  Fact 0.325: Chaffee, Mrs. Carrie Constance (née Toogood) boarded at Southampton.
  Fact 0.326: Chaffee, Mrs. Carrie Constance (née Toogood)'s destination is Amenia, North Dakota, US.
  Fact 0.327: Chambers, Mr. Norman Campbell was 27 years old.
  Fact 0.328: Chambers, Mr. Norman Campbell is from New York City.
  Fact 0.329: Chambers, Mr. Norman Campbell boarded at Southampton.
  Fact 0.330: Chambers, Mr. Norman Campbell's destination is New York City.
  Fact 0.331: Chambers, Mr. Norman Campbell has 5 in the lifeboat or body column.
  Fact 0.332: Chambers, Mrs. Bertha (née Griggs) was 32 years old.
  Fact 0.333: Chambers, Mrs. Bertha (née Griggs) is from New York City.
  Fact 0.334: Chambers, Mrs. Bertha (née Griggs) boarded at Southampton.
  Fact 0.335: Chambers, Mrs. Bertha (née Griggs)'s destination is New York City.
  Fact 0.336: Chambers, Mrs. Bertha (née Griggs) has 5 in the lifeboat or body column.
  Fact 0.337: Cherry, Miss Gladys was 30 years old.
  Fact 0.338: Cherry, Miss Gladys is from London, England, UK.
  Fact 0.339: Cherry, Miss Gladys boarded at Southampton.
  Fact 0.340: Cherry, Miss Gladys's destination is Vancouver, British Columbia, Canada.
  Fact 0.341: Cherry, Miss Gladys has 8 in the lifeboat or body column.
  Fact 0.342: Chevré, Mr. Paul Romaine Marie Léonce was 45 years old.
  Fact 0.343: Chevré, Mr. Paul Romaine Marie Léonce is from Paris, France.
  Fact 0.344: Chevré, Mr. Paul Romaine Marie Léonce boarded at Cherbourg.
  Fact 0.345: Chevré, Mr. Paul Romaine Marie Léonce's destination is Ottawa, Ontario, Canada.
  Fact 0.346: Chevré, Mr. Paul Romaine Marie Léonce has 7 in the lifeboat or body column.
  Fact 0.347: Chibnall, Mrs. Edith Martha Bowerman (née Barber) was 48 years old.
  Fact 0.348: Chibnall, Mrs. Edith Martha Bowerman (née Barber) is from St. Leonards-on-Sea, East Sussex, England, UK.
  Fact 0.349: Chibnall, Mrs. Edith Martha Bowerman (née Barber) boarded at Southampton.
  Fact 0.350: Chibnall, Mrs. Edith Martha Bowerman (née Barber)'s destination is New York City.
  Fact 0.351: Chibnall, Mrs. Edith Martha Bowerman (née Barber) has 6 in the lifeboat or body column.
  Fact 0.352: Chisholm, Mr. Roderick Robert Crispin was 40 years old.
  Fact 0.353: Chisholm, Mr. Roderick Robert Crispin is from Belfast, Ireland, UK.
  Fact 0.354: Chisholm, Mr. Roderick Robert Crispin boarded at Belfast.
  Fact 0.355: Chisholm, Mr. Roderick Robert Crispin's destination is New York City.
  Fact 0.356: Clark, Mr. Walter Miller was 27 years old.
  Fact 0.357: Clark, Mr. Walter Miller is from Los Angeles, California, US.
  Fact 0.358: Clark, Mr. Walter Miller boarded at Cherbourg.
  Fact 0.359: Clark, Mr. Walter Miller's destination is Los Angeles, California, US.
  Fact 0.360: Clark, Mrs. Virginia Estelle (née McDowell) was 26 years old.
  Fact 0.361: Clark, Mrs. Virginia Estelle (née McDowell) is from Los Angeles, California, US.
  Fact 0.362: Clark, Mrs. Virginia Estelle (née McDowell) boarded at Cherbourg.
  Fact 0.363: Clark, Mrs. Virginia Estelle (née McDowell)'s destination is Los Angeles, California, US.
  Fact 0.364: Clifford, Mr. George Quincy was 40 years old.
  Fact 0.365: Clifford, Mr. George Quincy is from Stoughton, Massachusetts, US.
  Fact 0.366: Clifford, Mr. George Quincy boarded at Southampton.
  Fact 0.367: Clifford, Mr. George Quincy's destination is Stoughton, Massachusetts.
  Fact 0.368: Colley, Mr. Edward Pomeroy was 37 years old.
  Fact 0.369: Colley, Mr. Edward Pomeroy is from Dublin, Ireland, UK.
  Fact 0.370: Colley, Mr. Edward Pomeroy boarded at Southampton.
  Fact 0.371: Colley, Mr. Edward Pomeroy's destination is Vancouver, British Columbia, Canada.
  Fact 0.372: Compton, Mrs. Mary Eliza (née Ingersoll) was 64 years old.
  Fact 0.373: Compton, Mrs. Mary Eliza (née Ingersoll) is from Lakewood, New Jersey, US.
  Fact 0.374: Compton, Mrs. Mary Eliza (née Ingersoll) boarded at Cherbourg.
  Fact 0.375: Compton, Mrs. Mary Eliza (née Ingersoll)'s destination is Lakewood, New Jersey, US.
  Fact 0.376: Compton, Mrs. Mary Eliza (née Ingersoll) has 14 in the lifeboat or body column.
  Fact 0.377: Compton, Miss Sara Rebecca was 39 years old.
  Fact 0.378: Compton, Miss Sara Rebecca is from Lakewood, New Jersey, US.
  Fact 0.379: Compton, Miss Sara Rebecca boarded at Cherbourg.
  Fact 0.380: Compton, Miss Sara Rebecca's destination is Lakewood, New Jersey, US.
  Fact 0.381: Compton, Miss Sara Rebecca has 14 in the lifeboat or body column.
  Fact 0.382: Compton, Mr. Alexander Taylor Jr. was 37 years old.
  Fact 0.383: Compton, Mr. Alexander Taylor Jr. is from Lakewood, New Jersey, US.
  Fact 0.384: Compton, Mr. Alexander Taylor Jr. boarded at Cherbourg.
  Fact 0.385: Compton, Mr. Alexander Taylor Jr.'s destination is Lakewood, New Jersey, US.
  Fact 0.386: Cornell, Mrs. Malvina Helen (née Lamson) was 55 years old.
  Fact 0.387: Cornell, Mrs. Malvina Helen (née Lamson) is from New York City.
  Fact 0.388: Cornell, Mrs. Malvina Helen (née Lamson) boarded at Southampton.
  Fact 0.389: Cornell, Mrs. Malvina Helen (née Lamson)'s destination is New York City.
  Fact 0.390: Crafton, Mr. John Bertram was 59 years old.
  Fact 0.391: Crafton, Mr. John Bertram is from Roachdale, Indiana, US.
  Fact 0.392: Crafton, Mr. John Bertram boarded at Southampton.
  Fact 0.393: Crafton, Mr. John Bertram's destination is Roachdale, Indiana, US.
  Fact 0.394: Crosby, Captain Edward was 70 years old.
  Fact 0.395: Crosby, Captain Edward is from Milwaukee, Wisconsin, US.
  Fact 0.396: Crosby, Captain Edward boarded at Southampton.
  Fact 0.397: Crosby, Captain Edward's destination is Milwaukee, Wisconsin, US.
  Fact 0.398: Crosby, Captain Edward has 269 in the lifeboat or body column.
  Fact 0.399: Crosby, Mrs. Catherine Elizabeth (née Halstead) was 64 years old.
  Fact 0.400: Crosby, Mrs. Catherine Elizabeth (née Halstead) is from Milwaukee, Wisconsin, US.
  Fact 0.401: Crosby, Mrs. Catherine Elizabeth (née Halstead) boarded at Southampton.
  Fact 0.402: Crosby, Mrs. Catherine Elizabeth (née Halstead)'s destination is Milwaukee, Wisconsin, US.
  Fact 0.403: Crosby, Mrs. Catherine Elizabeth (née Halstead) has 7 in the lifeboat or body column.
  Fact 0.404: Crosby, Miss Harriette Rebecca was 39 years old.
  Fact 0.405: Crosby, Miss Harriette Rebecca is from Milwaukee, Wisconsin, US.
  Fact 0.406: Crosby, Miss Harriette Rebecca boarded at Southampton.
  Fact 0.407: Crosby, Miss Harriette Rebecca's destination is Milwaukee, Wisconsin, US.
  Fact 0.408: Crosby, Miss Harriette Rebecca has 7 in the lifeboat or body column.
  Fact 0.409: Cumings, Mr. John Bradley was 39 years old.
  Fact 0.410: Cumings, Mr. John Bradley is from New York City.
  Fact 0.411: Cumings, Mr. John Bradley boarded at Cherbourg.
  Fact 0.412: Cumings, Mr. John Bradley's destination is New York City.
  Fact 0.413: Cumings, Mrs. Florence Briggs (née Thayer) was 35 years old.
  Fact 0.414: Cumings, Mrs. Florence Briggs (née Thayer) is from New York City.
  Fact 0.415: Cumings, Mrs. Florence Briggs (née Thayer) boarded at Cherbourg.
  Fact 0.416: Cumings, Mrs. Florence Briggs (née Thayer)'s destination is New York, US.
  Fact 0.417: Daly, Mr. Peter Dennis was 51 years old.
  Fact 0.418: Daly, Mr. Peter Dennis is from Lima, Peru.
  Fact 0.419: Daly, Mr. Peter Dennis boarded at Southampton.
  Fact 0.420: Daly, Mr. Peter Dennis's destination is Lima, Peru.
  Fact 0.421: Daniel, Mr. Robert Williams was 27 years old.
  Fact 0.422: Daniel, Mr. Robert Williams is from Philadelphia, Pennsylvania, US.
  Fact 0.423: Daniel, Mr. Robert Williams boarded at Southampton.
  Fact 0.424: Daniel, Mr. Robert Williams's destination is Philadelphia, Pennsylvania, US.
  Fact 0.425: Davidson, Mr. Thornton was 31 years old.
  Fact 0.426: Davidson, Mr. Thornton is from Montreal, Quebec, Canada.
  Fact 0.427: Davidson, Mr. Thornton boarded at Southampton.
  Fact 0.428: Davidson, Mr. Thornton's destination is Montreal, Quebec, Canada.
  Fact 0.429: Davidson, Mrs. Orian (née Hays) was 27 years old.
  Fact 0.430: Davidson, Mrs. Orian (née Hays) is from Montreal, Quebec, Canada.
  Fact 0.431: Davidson, Mrs. Orian (née Hays) boarded at Southampton.
  Fact 0.432: Davidson, Mrs. Orian (née Hays)'s destination is Montreal, Quebec, Canada.
  Fact 0.433: Dick, Mr. Albert Adrian was 31 years old.
  Fact 0.434: Dick, Mr. Albert Adrian is from Calgary, Alberta, Canada.
  Fact 0.435: Dick, Mr. Albert Adrian boarded at Southampton.
  Fact 0.436: Dick, Mr. Albert Adrian's destination is Calgary, Alberta, Canada.
  Fact 0.437: Dick, Mrs. Vera (née Gillespie) was 17 years old.
  Fact 0.438: Dick, Mrs. Vera (née Gillespie) is from Calgary, Alberta, Canada.
  Fact 0.439: Dick, Mrs. Vera (née Gillespie) boarded at Southampton.
  Fact 0.440: Dick, Mrs. Vera (née Gillespie)'s destination is Calgary, Alberta, Canada.
  Fact 0.441: Dineston, Mrs. Nancy May Bowles was 28 years old.
  Fact 0.442: Dineston, Mrs. Nancy May Bowles is from London, England, UK.
  Fact 0.443: Dineston, Mrs. Nancy May Bowles boarded at Southampton.
  Fact 0.444: Dineston, Mrs. Nancy May Bowles's destination is New York City, USA.
  Fact 0.445: Dineston, George Samuel was 30 years old.
  Fact 0.446: Dineston, George Samuel is from London, England, UK.
  Fact 0.447: Dineston, George Samuel boarded at Southampton.
  Fact 0.448: Dineston, George Samuel's destination is New York City, USA.
  Fact 0.449: Dodge, Dr. Washington was 52 years old.
  Fact 0.450: Dodge, Dr. Washington is from San Francisco, California, US.
  Fact 0.451: Dodge, Dr. Washington boarded at Southampton.
  Fact 0.452: Dodge, Dr. Washington's destination is San Francisco, California, US.
  Fact 0.453: Dodge, Dr. Washington has 13 in the lifeboat or body column.
  Fact 0.454: Dodge, Mrs. Ruth (née Vidaver) was 34 years old.
  Fact 0.455: Dodge, Mrs. Ruth (née Vidaver) is from San Francisco, California, US.
  Fact 0.456: Dodge, Mrs. Ruth (née Vidaver) boarded at Southampton.
  Fact 0.457: Dodge, Mrs. Ruth (née Vidaver)'s destination is San Francisco, California, US.
  Fact 0.458: Dodge, Mrs. Ruth (née Vidaver) has 5 in the lifeboat or body column.
  Fact 0.459: Dodge, Master Washington, Jr. is from San Francisco, California, US.
  Fact 0.460: Dodge, Master Washington, Jr. boarded at Southampton.
  Fact 0.461: Dodge, Master Washington, Jr.'s destination is San Francisco, California, US.
  Fact 0.462: Dodge, Master Washington, Jr. has 5 in the lifeboat or body column.
  Fact 0.463: Douglas, Mr. Walter Donald was 50 years old.
  Fact 0.464: Douglas, Mr. Walter Donald is from Minneapolis, Minnesota, US.
  Fact 0.465: Douglas, Mr. Walter Donald boarded at Cherbourg.
  Fact 0.466: Douglas, Mr. Walter Donald's destination is Minneapolis, Minnesota, US.
  Fact 0.467: Douglas, Mr. Walter Donald has 62 in the lifeboat or body column.
  Fact 0.468: Douglas, Mrs. Mahala (née Dutton) was 48 years old.
  Fact 0.469: Douglas, Mrs. Mahala (née Dutton) is from Minneapolis, Minnesota, US.
  Fact 0.470: Douglas, Mrs. Mahala (née Dutton) boarded at Cherbourg.
  Fact 0.471: Douglas, Mrs. Mahala (née Dutton)'s destination is Minneapolis, Minnesota, US.
  Fact 0.472: Miss Berthe Leroy was 27 years old.
  Fact 0.473: Miss Berthe Leroy is listed as the maid.
  Fact 0.474: Miss Berthe Leroy is from Minneapolis, Minnesota, US.
  Fact 0.475: Miss Berthe Leroy boarded at Cherbourg.
  Fact 0.476: Miss Berthe Leroy's destination is Minneapolis, Minnesota, US.
  Fact 0.477: Douglas, Mrs. Mary Hélène (née Baxter) was 27 years old.
  Fact 0.478: Douglas, Mrs. Mary Hélène (née Baxter) is from Montreal, Quebec, Canada.
  Fact 0.479: Douglas, Mrs. Mary Hélène (née Baxter) boarded at Cherbourg.
  Fact 0.480: Douglas, Mrs. Mary Hélène (née Baxter)'s destination is Montreal, Quebec, Canada.
  Fact 0.481: Douglas, Mrs. Mary Hélène (née Baxter) has 6 in the lifeboat or body column.
  Fact 0.482: Duff Gordon, Sir Cosmo Edmund was 49 years old.
  Fact 0.483: Duff Gordon, Sir Cosmo Edmund is from London, England, UK.
  Fact 0.484: Duff Gordon, Sir Cosmo Edmund boarded at Cherbourg.
  Fact 0.485: Duff Gordon, Sir Cosmo Edmund's destination is New York City.
  Fact 0.486: Duff Gordon, Lucy Christiana, Lady (née Sutherland) was 48 years old.
  Fact 0.487: Duff Gordon, Lucy Christiana, Lady (née Sutherland) is from London, England, UK.
  Fact 0.488: Duff Gordon, Lucy Christiana, Lady (née Sutherland) boarded at Cherbourg.
  Fact 0.489: Duff Gordon, Lucy Christiana, Lady (née Sutherland)'s destination is New York City.
  Fact 0.490: Miss Laura Mabel Francatelli was 31 years old.
  Fact 0.491: Miss Laura Mabel Francatelli is listed as the secretary.
  Fact 0.492: Miss Laura Mabel Francatelli is from London, England, UK.
  Fact 0.493: Miss Laura Mabel Francatelli boarded at Cherbourg.
  Fact 0.494: Miss Laura Mabel Francatelli's destination is New York City.
  Fact 0.495: Dulles, Mr. William Crothers was 39 years old.
  Fact 0.496: Dulles, Mr. William Crothers is from Philadelphia, Pennsylvania, US.
  Fact 0.497: Dulles, Mr. William Crothers boarded at Cherbourg.
  Fact 0.498: Dulles, Mr. William Crothers's destination is Philadelphia, Pennsylvania, US.
  Fact 0.499: Dulles, Mr. William Crothers has 133 in the lifeboat or body column.
  Fact 0.500: Eustis, Miss Elizabeth Mussey was 54 years old.
  Fact 0.501: Eustis, Miss Elizabeth Mussey is from Brookline, Massachusetts, US.
  Fact 0.502: Eustis, Miss Elizabeth Mussey boarded at Cherbourg.
  Fact 0.503: Eustis, Miss Elizabeth Mussey's destination is Brookline, Massachusetts, US.
  Fact 0.504: Evans, Miss Edith Corse was 36 years old.
  Fact 0.505: Evans, Miss Edith Corse is from New York City.
  Fact 0.506: Evans, Miss Edith Corse boarded at Cherbourg.
  Fact 0.507: Evans, Miss Edith Corse's destination is New York City.
  Fact 0.508: Flegenheim, Mrs. Antoinette (née Wendt) was 48 years old.
  Fact 0.509: Flegenheim, Mrs. Antoinette (née Wendt) is from New York City.
  Fact 0.510: Flegenheim, Mrs. Antoinette (née Wendt) boarded at Cherbourg.
  Fact 0.511: Flegenheim, Mrs. Antoinette (née Wendt)'s destination is New York City.
  Fact 0.512: Flegenheim, Mrs. Antoinette (née Wendt) has 7 in the lifeboat or body column.
  Fact 0.513: Flynn, Mr. John Irwin was 36 years old.
  Fact 0.514: Flynn, Mr. John Irwin is from Brooklyn, New York, US.
  Fact 0.515: Flynn, Mr. John Irwin boarded at Southampton.
  Fact 0.516: Flynn, Mr. John Irwin's destination is Brooklyn, New York, US.
  Fact 0.517: Flynn, Mr. John Irwin has 5 in the lifeboat or body column.
  Fact 0.518: Foreman, Mr. Benjamin Laventall was 30 years old.
  Fact 0.519: Foreman, Mr. Benjamin Laventall is from New York City.
  Fact 0.520: Foreman, Mr. Benjamin Laventall boarded at Southampton.
  Fact 0.521: Foreman, Mr. Benjamin Laventall's destination is New York City.
  Fact 0.522: Fortune, Mr. Mark was 64 years old.
  Fact 0.523: Fortune, Mr. Mark is from Winnipeg, Manitoba, Canada.
  Fact 0.524: Fortune, Mr. Mark boarded at Southampton.
  Fact 0.525: Fortune, Mr. Mark's destination is Winnipeg, Manitoba, Canada.
  Fact 0.526: Fortune, Mrs. Mary (née McDougald) was 60 years old.
  Fact 0.527: Fortune, Mrs. Mary (née McDougald) is from Winnipeg, Manitoba, Canada.
  Fact 0.528: Fortune, Mrs. Mary (née McDougald) boarded at Southampton.
  Fact 0.529: Fortune, Mrs. Mary (née McDougald)'s destination is Winnipeg, Manitoba, Canada.
  Fact 0.530: Fortune, Mrs. Mary (née McDougald) has 10 in the lifeboat or body column.
  Fact 0.531: Fortune, Miss Ethel Flora was 28 years old.
  Fact 0.532: Fortune, Miss Ethel Flora is from Winnipeg, Manitoba, Canada.
  Fact 0.533: Fortune, Miss Ethel Flora boarded at Southampton.
  Fact 0.534: Fortune, Miss Ethel Flora's destination is Winnipeg, Manitoba, Canada.
  Fact 0.535: Fortune, Miss Ethel Flora has 10 in the lifeboat or body column.
  Fact 0.536: Fortune, Miss Alice Elizabeth was 24 years old.
  Fact 0.537: Fortune, Miss Alice Elizabeth is from Winnipeg, Manitoba, Canada.
  Fact 0.538: Fortune, Miss Alice Elizabeth boarded at Southampton.
  Fact 0.539: Fortune, Miss Alice Elizabeth's destination is Winnipeg, Manitoba, Canada.
  Fact 0.540: Fortune, Miss Alice Elizabeth has 10 in the lifeboat or body column.
  Fact 0.541: Fortune, Miss Mabel Helen was 23 years old.
  Fact 0.542: Fortune, Miss Mabel Helen is from Winnipeg, Manitoba, Canada.
  Fact 0.543: Fortune, Miss Mabel Helen boarded at Southampton.
  Fact 0.544: Fortune, Miss Mabel Helen's destination is Winnipeg, Manitoba, Canada.
  Fact 0.545: Fortune, Miss Mabel Helen has 10 in the lifeboat or body column.
  Fact 0.546: Fortune, Mr. Charles Alexander was 19 years old.
  Fact 0.547: Fortune, Mr. Charles Alexander is from Winnipeg, Manitoba, Canada.
  Fact 0.548: Fortune, Mr. Charles Alexander boarded at Southampton.
  Fact 0.549: Fortune, Mr. Charles Alexander's destination is Winnipeg, Manitoba, Canada.
  Fact 0.550: Franklin, Mr. Thomas Parnham was 37 years old.
  Fact 0.551: Franklin, Mr. Thomas Parnham is from London, England, UK.
  Fact 0.552: Franklin, Mr. Thomas Parnham boarded at Southampton.
  Fact 0.553: Franklin, Mr. Thomas Parnham's destination is New York City.
  Fact 0.554: Frauenthal, Dr. Henry William was 49 years old.
  Fact 0.555: Frauenthal, Dr. Henry William is from London, England, UK.
  Fact 0.556: Frauenthal, Dr. Henry William boarded at Southampton.
  Fact 0.557: Frauenthal, Dr. Henry William's destination is New York City.
  Fact 0.558: Frauenthal, Dr. Henry William has 5 in the lifeboat or body column.
  Fact 0.559: Frauenthal, Mrs. Clara (née Heinsheimer) was 42 years old.
  Fact 0.560: Frauenthal, Mrs. Clara (née Heinsheimer) is from London, England, UK.
  Fact 0.561: Frauenthal, Mrs. Clara (née Heinsheimer) boarded at Southampton.
  Fact 0.562: Frauenthal, Mrs. Clara (née Heinsheimer)'s destination is New York City.
  Fact 0.563: Frauenthal, Mrs. Clara (née Heinsheimer) has 5 in the lifeboat or body column.
  Fact 0.564: Frauenthal, Mr. Isaac Gerald was 43 years old.
  Fact 0.565: Frauenthal, Mr. Isaac Gerald is from London, England, UK.
  Fact 0.566: Frauenthal, Mr. Isaac Gerald boarded at Southampton.
  Fact 0.567: Frauenthal, Mr. Isaac Gerald's destination is New York City.
  Fact 0.568: Frauenthal, Mr. Isaac Gerald has 5 in the lifeboat or body column.
  Fact 0.569: Frölicher, Mr. Maximilian Josef was 60 years old.
  Fact 0.570: Frölicher, Mr. Maximilian Josef is from Zürich, Switzerland.
  Fact 0.571: Frölicher, Mr. Maximilian Josef boarded at Cherbourg.
  Fact 0.572: Frölicher, Mr. Maximilian Josef's destination is New York City.
  Fact 0.573: Frölicher, Mr. Maximilian Josef has 5 in the lifeboat or body column.
  Fact 0.574: Frölicher, Mrs. Margaretha Emerentia (née Stehli) was 48 years old.
  Fact 0.575: Frölicher, Mrs. Margaretha Emerentia (née Stehli) is from Zürich, Switzerland.
  Fact 0.576: Frölicher, Mrs. Margaretha Emerentia (née Stehli) boarded at Cherbourg.
  Fact 0.577: Frölicher, Mrs. Margaretha Emerentia (née Stehli)'s destination is New York City.
  Fact 0.578: Frölicher, Mrs. Margaretha Emerentia (née Stehli) has 5 in the lifeboat or body column.
  Fact 0.579: Frölicher-Stehli, Miss Hedwig Margaritha was 22 years old.
  Fact 0.580: Frölicher-Stehli, Miss Hedwig Margaritha is from Zürich, Switzerland.
  Fact 0.581: Frölicher-Stehli, Miss Hedwig Margaritha boarded at Cherbourg.
  Fact 0.582: Frölicher-Stehli, Miss Hedwig Margaritha's destination is New York City.
  Fact 0.583: Frölicher-Stehli, Miss Hedwig Margaritha has 5 in the lifeboat or body column.
  Fact 0.584: Futrelle, Mr. Jacques Heath was 37 years old.
  Fact 0.585: Futrelle, Mr. Jacques Heath is from Scituate, Massachusetts, US.
  Fact 0.586: Futrelle, Mr. Jacques Heath boarded at Southampton.
  Fact 0.587: Futrelle, Mr. Jacques Heath's destination is Scituate, Massachusetts, US.
  Fact 0.588: Futrelle, Mrs. Lily May (née Peel) was 35 years old.
  Fact 0.589: Futrelle, Mrs. Lily May (née Peel) is from Scituate, Massachusetts, US.
  Fact 0.590: Futrelle, Mrs. Lily May (née Peel) boarded at Southampton.
  Fact 0.591: Futrelle, Mrs. Lily May (née Peel)'s destination is Scituate, Massachusetts, US.
  Fact 0.592: Gee, Mr. Arthur H. was 47 years old.
  Fact 0.593: Gee, Mr. Arthur H. is from St. Annes-on-Sea, Lancashire, England, UK.
  Fact 0.594: Gee, Mr. Arthur H. boarded at Southampton.
  Fact 0.595: Gee, Mr. Arthur H.'s destination is Mexico City, Mexico.
  Fact 0.596: Gee, Mr. Arthur H. has 275 in the lifeboat or body column.
  Fact 0.597: Gibson, Mrs. Pauline Caroline (née Boeson) was 44 years old.
  Fact 0.598: Gibson, Mrs. Pauline Caroline (née Boeson) is from New York City.
  Fact 0.599: Gibson, Mrs. Pauline Caroline (née Boeson) boarded at Cherbourg.
  Fact 0.600: Gibson, Mrs. Pauline Caroline (née Boeson)'s destination is New York City.
  Fact 0.601: Gibson, Mrs. Pauline Caroline (née Boeson) has 7 in the lifeboat or body column.
  Fact 0.602: Gibson, Miss Dorothy Winifred was 22 years old.
  Fact 0.603: Gibson, Miss Dorothy Winifred is from New York City.
  Fact 0.604: Gibson, Miss Dorothy Winifred boarded at Cherbourg.
  Fact 0.605: Gibson, Miss Dorothy Winifred's destination is New York City.
  Fact 0.606: Gibson, Miss Dorothy Winifred has 7 in the lifeboat or body column.
  Fact 0.607: Goldenberg, Mr. Samuel L. was 47 years old.
  Fact 0.608: Goldenberg, Mr. Samuel L. is from Paris, France.
  Fact 0.609: Goldenberg, Mr. Samuel L. boarded at Cherbourg.
  Fact 0.610: Goldenberg, Mr. Samuel L.'s destination is New York City.
  Fact 0.611: Goldenberg, Mr. Samuel L. has 5 in the lifeboat or body column.
  Fact 0.612: Goldenberg, Mrs. Nella (née Wiggins) was 40 years old.
  Fact 0.613: Goldenberg, Mrs. Nella (née Wiggins) is from Paris, France.
  Fact 0.614: Goldenberg, Mrs. Nella (née Wiggins) boarded at Cherbourg.
  Fact 0.615: Goldenberg, Mrs. Nella (née Wiggins)'s destination is New York City.
  Fact 0.616: Goldenberg, Mrs. Nella (née Wiggins) has 5 in the lifeboat or body column.
  Fact 0.617: Goldschmidt, Mr. George B. was 71 years old.
  Fact 0.618: Goldschmidt, Mr. George B. is from New York City.
  Fact 0.619: Goldschmidt, Mr. George B. boarded at Cherbourg.
  Fact 0.620: Goldschmidt, Mr. George B.'s destination is New York City.
  Fact 0.621: Gracie IV, Colonel Archibald was 53 years old.
  Fact 0.622: Gracie IV, Colonel Archibald is from Washington, D.C., US.
  Fact 0.623: Gracie IV, Colonel Archibald boarded at Southampton.
  Fact 0.624: Gracie IV, Colonel Archibald's destination is Washington, D.C., US.
  Fact 0.625: Graham, Mr. George Edward was 38 years old.
  Fact 0.626: Graham, Mr. George Edward is from Winnipeg, Manitoba, Canada.
  Fact 0.627: Graham, Mr. George Edward boarded at Southampton.
  Fact 0.628: Graham, Mr. George Edward's destination is Winnipeg, Manitoba, Canada.
  Fact 0.629: Graham, Mr. George Edward has 147 in the lifeboat or body column.
  Fact 0.630: Graham, Mrs. Edith Ware (née Junkins) was 59 years old.
  Fact 0.631: Graham, Mrs. Edith Ware (née Junkins) is from Greenwich, Connecticut, US.
  Fact 0.632: Graham, Mrs. Edith Ware (née Junkins) boarded at Southampton.
  Fact 0.633: Graham, Mrs. Edith Ware (née Junkins)'s destination is Greenwich, Connecticut, US.
  Fact 0.634: Graham, Miss Margaret Edith was 19 years old.
  Fact 0.635: Graham, Miss Margaret Edith is from Greenwich, Connecticut, US.
  Fact 0.636: Graham, Miss Margaret Edith boarded at Southampton.
  Fact 0.637: Graham, Miss Margaret Edith's destination is Greenwich, Connecticut, US.
  Fact 0.638: Miss Elizabeth Weed Shutes was 40 years old.
  Fact 0.639: Miss Elizabeth Weed Shutes is listed as the governess.
  Fact 0.640: Miss Elizabeth Weed Shutes is from Greenwich, Connecticut, US.
  Fact 0.641: Miss Elizabeth Weed Shutes boarded at Southampton.
  Fact 0.642: Miss Elizabeth Weed Shutes's destination is Greenwich, Connecticut, US.
  Fact 0.643: Greenfield, Mrs. Blanche (née Strouse) was 45 years old.
  Fact 0.644: Greenfield, Mrs. Blanche (née Strouse) is from New York City.
  Fact 0.645: Greenfield, Mrs. Blanche (née Strouse) boarded at Cherbourg.
  Fact 0.646: Greenfield, Mrs. Blanche (née Strouse)'s destination is New York City.
  Fact 0.647: Greenfield, Mrs. Blanche (née Strouse) has 7 in the lifeboat or body column.
  Fact 0.648: Greenfield, Mr. William Bertram was 23 years old.
  Fact 0.649: Greenfield, Mr. William Bertram is from New York City.
  Fact 0.650: Greenfield, Mr. William Bertram boarded at Cherbourg.
  Fact 0.651: Greenfield, Mr. William Bertram's destination is New York City.
  Fact 0.652: Greenfield, Mr. William Bertram has 7 in the lifeboat or body column.
  Fact 0.653: Guggenheim, Mr. Benjamin was 46 years old.
  Fact 0.654: Guggenheim, Mr. Benjamin is from Paris, France.
  Fact 0.655: Guggenheim, Mr. Benjamin boarded at Cherbourg.
  Fact 0.656: Guggenheim, Mr. Benjamin's destination is New York City.
  Fact 0.657: Mr. Victor Giglio was 24 years old.
  Fact 0.658: Mr. Victor Giglio is listed as the valet.
  Fact 0.659: Mr. Victor Giglio is from Paris, France.
  Fact 0.660: Mr. Victor Giglio boarded at Cherbourg.
  Fact 0.661: Mr. Victor Giglio's destination is New York City.
  Fact 0.662: Mr. René Pernot was 39 years old.
  Fact 0.663: Mr. René Pernot is listed as the chauffeur.
  Fact 0.664: Mr. René Pernot is from Paris, France.
  Fact 0.665: Mr. René Pernot boarded at Cherbourg.
  Fact 0.666: Mr. René Pernot's destination is New York City.
  Fact 0.667: Harder, Mr. George Achilles was 25 years old.
  Fact 0.668: Harder, Mr. George Achilles is from New York City.
  Fact 0.669: Harder, Mr. George Achilles boarded at Cherbourg.
  Fact 0.670: Harder, Mr. George Achilles's destination is New York City.
  Fact 0.671: Harder, Mr. George Achilles has 5 in the lifeboat or body column.
  Fact 0.672: Harder, Mrs. Dorothy (née Annan) was 21 years old.
  Fact 0.673: Harder, Mrs. Dorothy (née Annan) is from New York City.
  Fact 0.674: Harder, Mrs. Dorothy (née Annan) boarded at Cherbourg.
  Fact 0.675: Harder, Mrs. Dorothy (née Annan)'s destination is New York City.
  Fact 0.676: Harder, Mrs. Dorothy (née Annan) has 5 in the lifeboat or body column.
  Fact 0.677: Harper, Mr. Henry Sleeper was 48 years old.
  Fact 0.678: Harper, Mr. Henry Sleeper is from New York City.
  Fact 0.679: Harper, Mr. Henry Sleeper boarded at Cherbourg.
  Fact 0.680: Harper, Mr. Henry Sleeper's destination is New York City.
  Fact 0.681: Mr. Hammad Hassab was 27 years old.
  Fact 0.682: Mr. Hammad Hassab is listed as the dragoman.
  Fact 0.683: Mr. Hammad Hassab is from Cairo, Egypt.
  Fact 0.684: Mr. Hammad Hassab boarded at Cherbourg.
  Fact 0.685: Mr. Hammad Hassab's destination is New York City.
  Fact 0.686: Harper, Mrs. Myna (née Haxtun) was 49 years old.
  Fact 0.687: Harper, Mrs. Myna (née Haxtun) is from New York City.
  Fact 0.688: Harper, Mrs. Myna (née Haxtun) boarded at Cherbourg.
  Fact 0.689: Harper, Mrs. Myna (née Haxtun)'s destination is New York City.
  Fact 0.690: Harris, Mr. Henry Birkhardt was 45 years old.
  Fact 0.691: Harris, Mr. Henry Birkhardt is from New York City.
  Fact 0.692: Harris, Mr. Henry Birkhardt boarded at Southampton.
  Fact 0.693: Harris, Mr. Henry Birkhardt's destination is New York City.
  Fact 0.694: Harris, Mrs. Irene (née Wallach) was 35 years old.
  Fact 0.695: Harris, Mrs. Irene (née Wallach) is from New York City.
  Fact 0.696: Harris, Mrs. Irene (née Wallach) boarded at Southampton.
  Fact 0.697: Harris, Mrs. Irene (née Wallach)'s destination is New York City.
  Fact 0.698: Hawksford, Mr. Walter James was 45 years old.
  Fact 0.699: Hawksford, Mr. Walter James is from Kingston, Surrey, England, UK.
  Fact 0.700: Hawksford, Mr. Walter James boarded at Southampton.
  Fact 0.701: Hawksford, Mr. Walter James's destination is New York City.
  Fact 0.702: Hays, Mr. Charles Melville was 55 years old.
  Fact 0.703: Hays, Mr. Charles Melville is from Montreal, Quebec, Canada.
  Fact 0.704: Hays, Mr. Charles Melville boarded at Southampton.
  Fact 0.705: Hays, Mr. Charles Melville's destination is Montreal, Quebec, Canada.
  Fact 0.706: Hays, Mr. Charles Melville has 307 in the lifeboat or body column.
  Fact 0.707: Mr. Vivian Ponsonby Payne was 22 years old.
  Fact 0.708: Mr. Vivian Ponsonby Payne is listed as the clerk.
  Fact 0.709: Mr. Vivian Ponsonby Payne is from Montreal, Quebec, Canada.
  Fact 0.710: Mr. Vivian Ponsonby Payne boarded at Southampton.
  Fact 0.711: Mr. Vivian Ponsonby Payne's destination is Montreal, Quebec, Canada.
  Fact 0.712: Hays, Mrs. Clara Jennings (née Grigg) was 52 years old.
  Fact 0.713: Hays, Mrs. Clara Jennings (née Grigg) is from Montreal, Quebec, Canada.
  Fact 0.714: Hays, Mrs. Clara Jennings (née Grigg) boarded at Southampton.
  Fact 0.715: Hays, Mrs. Clara Jennings (née Grigg)'s destination is Montreal, Quebec, Canada.
  Fact 0.716: Miss Mary Anne Perreault was 33 years old.
  Fact 0.717: Miss Mary Anne Perreault is listed as the maid.
  Fact 0.718: Miss Mary Anne Perreault is from Montreal, Quebec, Canada.
  Fact 0.719: Miss Mary Anne Perreault boarded at Southampton.
  Fact 0.720: Miss Mary Anne Perreault's destination is Montreal, Quebec, Canada.
  Fact 0.721: Hays, Miss Margaret Bechstein was 24 years old.
  Fact 0.722: Hays, Miss Margaret Bechstein is from New York City.
  Fact 0.723: Hays, Miss Margaret Bechstein boarded at Cherbourg.
  Fact 0.724: Hays, Miss Margaret Bechstein's destination is New York City.
  Fact 0.725: Hays, Miss Margaret Bechstein has 7 in the lifeboat or body column.
  Fact 0.726: Head, Mr. Christopher was 42 years old.
  Fact 0.727: Head, Mr. Christopher is from Chelsea, London, England, UK.
  Fact 0.728: Head, Mr. Christopher boarded at Southampton.
  Fact 0.729: Head, Mr. Christopher's destination is New York City.
  Fact 0.730: Hilliard, Mr. Herbert Henry was 44 years old.
  Fact 0.731: Hilliard, Mr. Herbert Henry is from Brighton, Massachusetts, US.
  Fact 0.732: Hilliard, Mr. Herbert Henry boarded at Southampton.
  Fact 0.733: Hilliard, Mr. Herbert Henry's destination is Brighton, Massachusetts, US.
  Fact 0.734: Hipkins, Mr. William Edward was 55 years old.
  Fact 0.735: Hipkins, Mr. William Edward is from Birmingham, West Midlands, England, UK.
  Fact 0.736: Hipkins, Mr. William Edward boarded at Southampton.
  Fact 0.737: Hipkins, Mr. William Edward's destination is New York City.
  Fact 0.738: Hippach, Mrs. Ida Sophia (née Fischer) was 44 years old.
  Fact 0.739: Hippach, Mrs. Ida Sophia (née Fischer) is from Chicago, Illinois, US.
  Fact 0.740: Hippach, Mrs. Ida Sophia (née Fischer) boarded at Cherbourg.
  Fact 0.741: Hippach, Mrs. Ida Sophia (née Fischer)'s destination is Chicago, Illinois, US.
  Fact 0.742: Hippach, Miss Jean Gertrude was 17 years old.
  Fact 0.743: Hippach, Miss Jean Gertrude is from Chicago, Illinois, US.
  Fact 0.744: Hippach, Miss Jean Gertrude boarded at Cherbourg.
  Fact 0.745: Hippach, Miss Jean Gertrude's destination is Chicago, Illinois, US.
  Fact 0.746: Hogeboom, Mrs. Anna Louisa (née Andrews) was 51 years old.
  Fact 0.747: Hogeboom, Mrs. Anna Louisa (née Andrews) is from Hudson, New York, US.
  Fact 0.748: Hogeboom, Mrs. Anna Louisa (née Andrews) boarded at Cherbourg.
  Fact 0.749: Hogeboom, Mrs. Anna Louisa (née Andrews)'s destination is Hudson, New York, US.
  Fact 0.750: Hogeboom, Mrs. Anna Louisa (née Andrews) has 10 in the lifeboat or body column.
  Fact 0.751: Holverson, Mr. Alexander Oskar was 42 years old.
  Fact 0.752: Holverson, Mr. Alexander Oskar is from New York City.
  Fact 0.753: Holverson, Mr. Alexander Oskar boarded at Southampton.
  Fact 0.754: Holverson, Mr. Alexander Oskar's destination is New York City.
  Fact 0.755: Holverson, Mr. Alexander Oskar has 38 in the lifeboat or body column.
  Fact 0.756: Holverson, Mrs. Mary Aline (née Towner) was 35 years old.
  Fact 0.757: Holverson, Mrs. Mary Aline (née Towner) is from New York City.
  Fact 0.758: Holverson, Mrs. Mary Aline (née Towner) boarded at Southampton.
  Fact 0.759: Holverson, Mrs. Mary Aline (née Towner)'s destination is New York City.
  Fact 0.760: Holverson, Mrs. Mary Aline (née Towner) has 8 in the lifeboat or body column.
  Fact 0.761: Homer, Mr. Harry (alias E. Haven) was 40 years old.
  Fact 0.762: Homer, Mr. Harry (alias E. Haven) is from Indianapolis, Indiana, US.
  Fact 0.763: Homer, Mr. Harry (alias E. Haven) boarded at Southampton.
  Fact 0.764: Homer, Mr. Harry (alias E. Haven)'s destination is Indianapolis, Indiana, US.
  Fact 0.765: Homer, Mr. Harry (alias E. Haven) has 15 in the lifeboat or body column.
  Fact 0.766: Hoyt, Mr. Frederick Maxfield was 35 years old.
  Fact 0.767: Hoyt, Mr. Frederick Maxfield is from New York City.
  Fact 0.768: Hoyt, Mr. Frederick Maxfield boarded at Southampton.
  Fact 0.769: Hoyt, Mr. Frederick Maxfield's destination is Stamford, Connecticut, US.
  Fact 0.770: Hoyt, Mrs. Jane Anne (née Forby) was 31 years old.
  Fact 0.771: Hoyt, Mrs. Jane Anne (née Forby) is from New York City.
  Fact 0.772: Hoyt, Mrs. Jane Anne (née Forby) boarded at Southampton.
  Fact 0.773: Hoyt, Mrs. Jane Anne (née Forby)'s destination is Stamford, Connecticut, US.
  Fact 0.774: Hoyt, Mr. William Fisher was 42 years old.
  Fact 0.775: Hoyt, Mr. William Fisher is from New York City.
  Fact 0.776: Hoyt, Mr. William Fisher boarded at Cherbourg.
  Fact 0.777: Hoyt, Mr. William Fisher's destination is New York City.
  Fact 0.778: Hoyt, Mr. William Fisher has 14 in the lifeboat or body column.
  Fact 0.779: Isham, Miss Ann Elizabeth was 50 years old.
  Fact 0.780: Isham, Miss Ann Elizabeth is from Chicago, Illinois, US.
  Fact 0.781: Isham, Miss Ann Elizabeth boarded at Cherbourg.
  Fact 0.782: Isham, Miss Ann Elizabeth's destination is Chicago, Illinois, US.
  Fact 0.783: Ismay, Mr. Joseph Bruce was 49 years old.
  Fact 0.784: Ismay, Mr. Joseph Bruce is from Liverpool, Merseyside, England, UK.
  Fact 0.785: Ismay, Mr. Joseph Bruce boarded at Southampton.
  Fact 0.786: Ismay, Mr. Joseph Bruce's destination is New York City.
  Fact 0.787: Mr. John Richard Fry was 39 years old.
  Fact 0.788: Mr. John Richard Fry is listed as the valet.
  Fact 0.789: Mr. John Richard Fry is from Liverpool, Merseyside, England, UK.
  Fact 0.790: Mr. John Richard Fry boarded at Southampton.
  Fact 0.791: Mr. John Richard Fry's destination is New York City.
  Fact 0.792: Mr. William Henry Harrison was 45 years old.
  Fact 0.793: Mr. William Henry Harrison is listed as the secretary.
  Fact 0.794: Mr. William Henry Harrison is from Wallasey, Merseyside, England, UK.
  Fact 0.795: Mr. William Henry Harrison boarded at Southampton.
  Fact 0.796: Mr. William Henry Harrison's destination is New York City.
  Fact 0.797: Mr. William Henry Harrison has 110 in the lifeboat or body column.
  Fact 0.798: Johnson, Mrs. Madeline Eliza (née Austin) was 40 years old.
  Fact 0.799: Johnson, Mrs. Madeline Eliza (née Austin) is from London, England, UK.
  Fact 0.800: Johnson, Mrs. Madeline Eliza (née Austin) boarded at Southampton.
  Fact 0.801: Johnson, Mrs. Madeline Eliza (née Austin)'s destination is New York, New York, US.
  Fact 0.802: Johnson, Mrs. Madeline Eliza (née Austin) has 6 in the lifeboat or body column.
  Fact 0.803: Johnson (later changed to Langston), Mr. Samuel James was 10 years old.
  Fact 0.804: Johnson (later changed to Langston), Mr. Samuel James is from London, England, UK.
  Fact 0.805: Johnson (later changed to Langston), Mr. Samuel James boarded at Southampton.
  Fact 0.806: Johnson (later changed to Langston), Mr. Samuel James's destination is New York, New York, US.
  Fact 0.807: Jones, Mr. Charles Cresson was 46 years old.
  Fact 0.808: Jones, Mr. Charles Cresson is from Bennington, Vermont, US.
  Fact 0.809: Jones, Mr. Charles Cresson boarded at Southampton.
  Fact 0.810: Jones, Mr. Charles Cresson's destination is Bennington, Vermont, US.
  Fact 0.811: Jones, Mr. Charles Cresson has 80 in the lifeboat or body column.
  Fact 0.812: Julian, Mr. Henry Forbes was 50 years old.
  Fact 0.813: Julian, Mr. Henry Forbes is from Torquay, Devon, England, UK.
  Fact 0.814: Julian, Mr. Henry Forbes boarded at Southampton.
  Fact 0.815: Julian, Mr. Henry Forbes's destination is San Francisco, US.
  Fact 0.816: Kent, Mr. Edward Austin was 58 years old.
  Fact 0.817: Kent, Mr. Edward Austin is from Buffalo, New York, US.
  Fact 0.818: Kent, Mr. Edward Austin boarded at Cherbourg.
  Fact 0.819: Kent, Mr. Edward Austin's destination is Buffalo, New York, US.
  Fact 0.820: Kent, Mr. Edward Austin has 258 in the lifeboat or body column.
  Fact 0.821: Kenyon, Mr. Frederick R. was 41 years old.
  Fact 0.822: Kenyon, Mr. Frederick R. is from Pittsburgh, Pennsylvania, US.
  Fact 0.823: Kenyon, Mr. Frederick R. boarded at Southampton.
  Fact 0.824: Kenyon, Mr. Frederick R.'s destination is Pittsburgh, Pennsylvania, US.
  Fact 0.825: Kenyon, Mrs. Marion (née Stauffer) was 31 years old.
  Fact 0.826: Kenyon, Mrs. Marion (née Stauffer) is from Pittsburgh, Pennsylvania, US.
  Fact 0.827: Kenyon, Mrs. Marion (née Stauffer) boarded at Southampton.
  Fact 0.828: Kenyon, Mrs. Marion (née Stauffer)'s destination is Pittsburgh, Pennsylvania, US.
  Fact 0.829: Kenyon, Mrs. Marion (née Stauffer) has 8 in the lifeboat or body column.
  Fact 0.830: Kimball, Mr. Edwin Nelson Jr. was 42 years old.
  Fact 0.831: Kimball, Mr. Edwin Nelson Jr. is from Boston, Massachusetts, US.
  Fact 0.832: Kimball, Mr. Edwin Nelson Jr. boarded at Southampton.
  Fact 0.833: Kimball, Mr. Edwin Nelson Jr.'s destination is Boston, Massachusetts, US.
  Fact 0.834: Kimball, Mr. Edwin Nelson Jr. has 5 in the lifeboat or body column.
  Fact 0.835: Kimball, Mrs. Gertrude (née Parsons) was 45 years old.
  Fact 0.836: Kimball, Mrs. Gertrude (née Parsons) is from Boston, Massachusetts, US.
  Fact 0.837: Kimball, Mrs. Gertrude (née Parsons) boarded at Southampton.
  Fact 0.838: Kimball, Mrs. Gertrude (née Parsons)'s destination is Boston, Massachusetts, US.
  Fact 0.839: Kimball, Mrs. Gertrude (née Parsons) has 5 in the lifeboat or body column.
  Fact 0.840: Klaber, Mr. Herman was 45 years old.
  Fact 0.841: Klaber, Mr. Herman is from Portland, Oregon, US.
  Fact 0.842: Klaber, Mr. Herman boarded at Southampton.
  Fact 0.843: Klaber, Mr. Herman's destination is Portland, Oregon, US.
  Fact 0.844: Langston, Ms. Maxine Jane was 25 years old.
  Fact 0.845: Langston, Ms. Maxine Jane is from Duncansville, Pennsylvania, US.
  Fact 0.846: Langston, Ms. Maxine Jane boarded at Southampton.
  Fact 0.847: Langston, Ms. Maxine Jane's destination is Duncansivlle, Pennsylvania, US.
  Fact 0.848: Lambert-Williams, Mr. Fletcher Fellows was 43 years old.
  Fact 0.849: Lambert-Williams, Mr. Fletcher Fellows is from London, England, UK.
  Fact 0.850: Lambert-Williams, Mr. Fletcher Fellows boarded at Southampton.
  Fact 0.851: Lambert-Williams, Mr. Fletcher Fellows's destination is Newark, New Jersey, US.
  Fact 0.852: Leader, Dr. Alice (née Farnham) was 49 years old.
  Fact 0.853: Leader, Dr. Alice (née Farnham) is from New York City.
  Fact 0.854: Leader, Dr. Alice (née Farnham) boarded at Southampton.
  Fact 0.855: Leader, Dr. Alice (née Farnham)'s destination is New York City.
  Fact 0.856: Leader, Dr. Alice (née Farnham) has 8 in the lifeboat or body column.
  Fact 0.857: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) was 27 years old.
  Fact 0.858: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) is from Vancouver, British Columbia, Canada.
  Fact 0.859: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) boarded at Southampton.
  Fact 0.860: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes)'s destination is Vancouver, British Columbia, Canada.
  Fact 0.861: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) has 8 in the lifeboat or body column.
  Fact 0.862: Miss Roberta Elizabeth Mary "Cissy" Maioni was 20 years old.
  Fact 0.863: Miss Roberta Elizabeth Mary "Cissy" Maioni is listed as the maid.
  Fact 0.864: Miss Roberta Elizabeth Mary "Cissy" Maioni is from Vancouver, British Columbia, Canada.
  Fact 0.865: Miss Roberta Elizabeth Mary "Cissy" Maioni boarded at Southampton.
  Fact 0.866: Miss Roberta Elizabeth Mary "Cissy" Maioni's destination is Vancouver, British Columbia, Canada.
  Fact 0.867: Miss Roberta Elizabeth Mary "Cissy" Maioni has 8 in the lifeboat or body column.
  Fact 0.868: Lewy, Mr. Ervin G. was 31 years old.
  Fact 0.869: Lewy, Mr. Ervin G. is from Chicago, Illinois, US.
  Fact 0.870: Lewy, Mr. Ervin G. boarded at Cherbourg.
  Fact 0.871: Lewy, Mr. Ervin G.'s destination is Chicago, Illinois, US.
  Fact 0.872: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) was 42 years old.
  Fact 0.873: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) is from Jordanstorp, Södermanland, Sweden.
  Fact 0.874: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) boarded at Southampton.
  Fact 0.875: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey)'s destination is New York City.
  Fact 0.876: Lindström, Mrs. Sigrid (née Posse) was 55 years old.
  Fact 0.877: Lindström, Mrs. Sigrid (née Posse) is from Stockholm, Sweden.
  Fact 0.878: Lindström, Mrs. Sigrid (née Posse) boarded at Cherbourg.
  Fact 0.879: Lindström, Mrs. Sigrid (née Posse)'s destination is New York City.
  Fact 0.880: Lindström, Mrs. Sigrid (née Posse) has 6 in the lifeboat or body column.
  Fact 0.881: Lines, Mrs. Elizabeth Lindsey (née James) was 50 years old.
  Fact 0.882: Lines, Mrs. Elizabeth Lindsey (née James) is from Paris, France.
  Fact 0.883: Lines, Mrs. Elizabeth Lindsey (née James) boarded at Cherbourg.
  Fact 0.884: Lines, Mrs. Elizabeth Lindsey (née James)'s destination is Hanover, New Hampshire, US.
  Fact 0.885: Lines, Mrs. Elizabeth Lindsey (née James) has 9 in the lifeboat or body column.
  Fact 0.886: Lines, Miss Mary Conover was 16 years old.
  Fact 0.887: Lines, Miss Mary Conover is from Paris, France.
  Fact 0.888: Lines, Miss Mary Conover boarded at Cherbourg.
  Fact 0.889: Lines, Miss Mary Conover's destination is Hanover, New Hampshire, US.
  Fact 0.890: Lines, Miss Mary Conover has 9 in the lifeboat or body column.
  Fact 0.891: Long, Mr. Milton Clyde was 29 years old.
  Fact 0.892: Long, Mr. Milton Clyde is from Springfield, Massachusetts, US.
  Fact 0.893: Long, Mr. Milton Clyde boarded at Southampton.
  Fact 0.894: Long, Mr. Milton Clyde's destination is Springfield, Massachusetts, US.
  Fact 0.895: Long, Mr. Milton Clyde has 126 in the lifeboat or body column.
  Fact 0.896: Longley, Miss Gretchen Fiske was 21 years old.
  Fact 0.897: Longley, Miss Gretchen Fiske is from Hudson, New York, US.
  Fact 0.898: Longley, Miss Gretchen Fiske boarded at Cherbourg.
  Fact 0.899: Longley, Miss Gretchen Fiske's destination is Hudson, New York, US.
  Fact 0.900: Longley, Miss Gretchen Fiske has 10 in the lifeboat or body column.
  Fact 0.901: Loring, Mr. Joseph Holland was 30 years old.
  Fact 0.902: Loring, Mr. Joseph Holland is from London, England, UK.
  Fact 0.903: Loring, Mr. Joseph Holland boarded at Southampton.
  Fact 0.904: Loring, Mr. Joseph Holland's destination is New York City.
  Fact 0.905: Madill, Miss Georgette Alexandra was 16 years old.
  Fact 0.906: Madill, Miss Georgette Alexandra is from St. Louis, Missouri, US.
  Fact 0.907: Madill, Miss Georgette Alexandra boarded at Southampton.
  Fact 0.908: Madill, Miss Georgette Alexandra's destination is St. Louis, Missouri, US.
  Fact 0.909: Maguire, Mr. John Edward was 30 years old.
  Fact 0.910: Maguire, Mr. John Edward is from Brockton, Massachusetts, US.
  Fact 0.911: Maguire, Mr. John Edward boarded at Southampton.
  Fact 0.912: Maguire, Mr. John Edward's destination is Brockton, Massachusetts, US.
  Fact 0.913: Maréchal, Mr. Pierre, Sr. was 28 years old.
  Fact 0.914: Maréchal, Mr. Pierre, Sr. is from Paris, France.
  Fact 0.915: Maréchal, Mr. Pierre, Sr. boarded at Cherbourg.
  Fact 0.916: Maréchal, Mr. Pierre, Sr.'s destination is New York City.
  Fact 0.917: Maréchal, Mr. Pierre, Sr. has 7 in the lifeboat or body column.
  Fact 0.918: Marvin, Mr. Daniel Warner was 18 years old.
  Fact 0.919: Marvin, Mr. Daniel Warner is from New York City.
  Fact 0.920: Marvin, Mr. Daniel Warner boarded at Southampton.
  Fact 0.921: Marvin, Mr. Daniel Warner's destination is New York City.
  Fact 0.922: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) was 18 years old.
  Fact 0.923: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) is from New York City.
  Fact 0.924: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) boarded at Southampton.
  Fact 0.925: Marvin, Mrs. Mary Graham Carmichael (née Farquarson)'s destination is New York City.
  Fact 0.926: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) has 10 in the lifeboat or body column.
  Fact 0.927: Mayné, Miss Bertha Antonine was 24 years old.
  Fact 0.928: Mayné, Miss Bertha Antonine is from Brussels, Belgium.
  Fact 0.929: Mayné, Miss Bertha Antonine boarded at Cherbourg.
  Fact 0.930: Mayné, Miss Bertha Antonine's destination is Montreal, Quebec, Canada.
  Fact 0.931: Mayné, Miss Bertha Antonine has 6 in the lifeboat or body column.
  Fact 0.932: McCaffry, Mr. Thomas Francis was 46 years old.
  Fact 0.933: McCaffry, Mr. Thomas Francis is from Vancouver, British Columbia, Canada.
  Fact 0.934: McCaffry, Mr. Thomas Francis boarded at Southampton.
  Fact 0.935: McCaffry, Mr. Thomas Francis's destination is Vancouver, British Columbia, Canada.
  Fact 0.936: McCaffry, Mr. Thomas Francis has 292 in the lifeboat or body column.
  Fact 0.937: McCarthy, Mr. Timothy J. was 54 years old.
  Fact 0.938: McCarthy, Mr. Timothy J. is from Dorchester, Massachusetts, US.
  Fact 0.939: McCarthy, Mr. Timothy J. boarded at Southampton.
  Fact 0.940: McCarthy, Mr. Timothy J.'s destination is Dorchester, Massachusetts, US.
  Fact 0.941: McCarthy, Mr. Timothy J. has 175 in the lifeboat or body column.
  Fact 0.942: McGough, Mr. James Robert was 35 years old.
  Fact 0.943: McGough, Mr. James Robert is from Philadelphia, Pennsylvania, US.
  Fact 0.944: McGough, Mr. James Robert boarded at Southampton.
  Fact 0.945: McGough, Mr. James Robert's destination is Philadelphia, Pennsylvania, US.
  Fact 0.946: McGough, Mr. James Robert has 7 in the lifeboat or body column.
  Fact 0.947: Meyer, Mr. Edgar Joseph was 28 years old.
  Fact 0.948: Meyer, Mr. Edgar Joseph is from New York City.
  Fact 0.949: Meyer, Mr. Edgar Joseph boarded at Cherbourg.
  Fact 0.950: Meyer, Mr. Edgar Joseph's destination is New York City.
  Fact 0.951: Meyer, Mrs. Leila (née Saks) was 25 years old.
  Fact 0.952: Meyer, Mrs. Leila (née Saks) is from New York City.
  Fact 0.953: Meyer, Mrs. Leila (née Saks) boarded at Cherbourg.
  Fact 0.954: Meyer, Mrs. Leila (née Saks)'s destination is New York City.
  Fact 0.955: Meyer, Mrs. Leila (née Saks) has 6 in the lifeboat or body column.
  Fact 0.956: Millet, Mr. Francis Davis was 65 years old.
  Fact 0.957: Millet, Mr. Francis Davis is from East Bridgewater, Massachusetts, US.
  Fact 0.958: Millet, Mr. Francis Davis boarded at Cherbourg.
  Fact 0.959: Millet, Mr. Francis Davis's destination is East Bridgewater, Massachusetts, US.
  Fact 0.960: Millet, Mr. Francis Davis has 249 in the lifeboat or body column.
  Fact 0.961: Minahan, Dr. William Edward was 44 years old.
  Fact 0.962: Minahan, Dr. William Edward is from Fond du Lac, Wisconsin, US.
  Fact 0.963: Minahan, Dr. William Edward boarded at Southampton.
  Fact 0.964: Minahan, Dr. William Edward's destination is Fond du Lac, Wisconsin, US.
  Fact 0.965: Minahan, Dr. William Edward has 230 in the lifeboat or body column.
  Fact 0.966: Minahan, Mrs. Lillian E. (née Thorpe) was 37 years old.
  Fact 0.967: Minahan, Mrs. Lillian E. (née Thorpe) is from Fond du Lac, Wisconsin, US.
  Fact 0.968: Minahan, Mrs. Lillian E. (née Thorpe) boarded at Southampton.
  Fact 0.969: Minahan, Mrs. Lillian E. (née Thorpe)'s destination is Fond du Lac, Wisconsin, US.
  Fact 0.970: Minahan, Mrs. Lillian E. (née Thorpe) has 14 in the lifeboat or body column.
  Fact 0.971: Minahan, Miss Daisy E. was 33 years old.
  Fact 0.972: Minahan, Miss Daisy E. is from Fond du Lac, Wisconsin, US.
  Fact 0.973: Minahan, Miss Daisy E. boarded at Southampton.
  Fact 0.974: Minahan, Miss Daisy E.'s destination is Fond du Lac, Wisconsin, US.
  Fact 0.975: Minahan, Miss Daisy E. has 14 in the lifeboat or body column.
  Fact 0.976: Mock, Mr. Philipp Edmund was 30 years old.
  Fact 0.977: Mock, Mr. Philipp Edmund is from New York City.
  Fact 0.978: Mock, Mr. Philipp Edmund boarded at Cherbourg.
  Fact 0.979: Mock, Mr. Philipp Edmund's destination is New York City.
  Fact 0.980: Mock, Mr. Philipp Edmund has 11 in the lifeboat or body column.
  Fact 0.981: Molson, Mr. Harry Markland was 55 years old.
  Fact 0.982: Molson, Mr. Harry Markland is from Montreal, Quebec, Canada.
  Fact 0.983: Molson, Mr. Harry Markland boarded at Southampton.
  Fact 0.984: Molson, Mr. Harry Markland's destination is Montreal, Quebec, Canada.
  Fact 0.985: Moore, Mr. Clarence Bloomfield was 47 years old.
  Fact 0.986: Moore, Mr. Clarence Bloomfield is from Washington, D.C., US.
  Fact 0.987: Moore, Mr. Clarence Bloomfield boarded at Southampton.
  Fact 0.988: Moore, Mr. Clarence Bloomfield's destination is Washington, D.C., US.
  Fact 0.989: Mr. Charles Henry Harrington was 37 years old.
  Fact 0.990: Mr. Charles Henry Harrington is listed as the valet.
  Fact 0.991: Mr. Charles Henry Harrington is from Washington, D.C., US.
  Fact 0.992: Mr. Charles Henry Harrington boarded at Southampton.
  Fact 0.993: Mr. Charles Henry Harrington's destination is Washington, D.C., US.
  Fact 0.994: Natsch, Mr. Charles was 36 years old.
  Fact 0.995: Natsch, Mr. Charles is from Brooklyn, New York, US.
  Fact 0.996: Natsch, Mr. Charles boarded at Cherbourg.
  Fact 0.997: Natsch, Mr. Charles's destination is Brooklyn, New York, US.
  Fact 0.998: Newell, Mr. Arthur Webster was 58 years old.
  Fact 0.999: Newell, Mr. Arthur Webster is from Lexington, Massachusetts, US.
  Fact 0.1000: Newell, Mr. Arthur Webster boarded at Cherbourg.
  Fact 0.1001: Newell, Mr. Arthur Webster's destination is Lexington, Massachusetts, US.
  Fact 0.1002: Newell, Mr. Arthur Webster has 122 in the lifeboat or body column.
  Fact 0.1003: Newell, Miss Madeleine was 31 years old.
  Fact 0.1004: Newell, Miss Madeleine is from Lexington, Massachusetts, US.
  Fact 0.1005: Newell, Miss Madeleine boarded at Cherbourg.
  Fact 0.1006: Newell, Miss Madeleine's destination is Lexington, Massachusetts, US.
  Fact 0.1007: Newell, Miss Madeleine has 6 in the lifeboat or body column.
  Fact 0.1008: Newell, Miss Marjorie Anne was 23 years old.
  Fact 0.1009: Newell, Miss Marjorie Anne is from Lexington, Massachusetts, US.
  Fact 0.1010: Newell, Miss Marjorie Anne boarded at Cherbourg.
  Fact 0.1011: Newell, Miss Marjorie Anne's destination is Lexington, Massachusetts, US.
  Fact 0.1012: Newell, Miss Marjorie Anne has 6 in the lifeboat or body column.
  Fact 0.1013: Newsom, Miss Helen Monypeny was 19 years old.
  Fact 0.1014: Newsom, Miss Helen Monypeny is from New York City.
  Fact 0.1015: Newsom, Miss Helen Monypeny boarded at Southampton.
  Fact 0.1016: Newsom, Miss Helen Monypeny's destination is New York City.
  Fact 0.1017: Newsom, Miss Helen Monypeny has 5 in the lifeboat or body column.
  Fact 0.1018: Nicholson, Mr. Arthur Ernest was 64 years old.
  Fact 0.1019: Nicholson, Mr. Arthur Ernest is from Shanklin, Isle of Wight, England, UK.
  Fact 0.1020: Nicholson, Mr. Arthur Ernest boarded at Southampton.
  Fact 0.1021: Nicholson, Mr. Arthur Ernest's destination is New York City.
  Fact 0.1022: Nicholson, Mr. Arthur Ernest has 263 in the lifeboat or body column.
  Fact 0.1023: Nourney, Mr. Alfred was 20 years old.
  Fact 0.1024: Nourney, Mr. Alfred is from Cologne, German Empire.
  Fact 0.1025: Nourney, Mr. Alfred boarded at Cherbourg.
  Fact 0.1026: Nourney, Mr. Alfred's destination is New York City.
  Fact 0.1027: Nourney, Mr. Alfred has 7 in the lifeboat or body column.
  Fact 0.1028: Omont, Mr. Alfred Fernand was 29 years old.
  Fact 0.1029: Omont, Mr. Alfred Fernand is from Le Havre, France.
  Fact 0.1030: Omont, Mr. Alfred Fernand boarded at Cherbourg.
  Fact 0.1031: Omont, Mr. Alfred Fernand's destination is New York City.
  Fact 0.1032: Omont, Mr. Alfred Fernand has 7 in the lifeboat or body column.
  Fact 0.1033: Ostby, Mr. Engelhart Cornelius was 64 years old.
  Fact 0.1034: Ostby, Mr. Engelhart Cornelius is from Providence, Rhode Island, US.
  Fact 0.1035: Ostby, Mr. Engelhart Cornelius boarded at Southampton.
  Fact 0.1036: Ostby, Mr. Engelhart Cornelius's destination is Providence, Rhode Island, US.
  Fact 0.1037: Ostby, Mr. Engelhart Cornelius has 234 in the lifeboat or body column.
  Fact 0.1038: Ostby, Miss Helene Ragnhild was 22 years old.
  Fact 0.1039: Ostby, Miss Helene Ragnhild is from Providence, Rhode Island, US.
  Fact 0.1040: Ostby, Miss Helene Ragnhild boarded at Southampton.
  Fact 0.1041: Ostby, Miss Helene Ragnhild's destination is Providence, Rhode Island, US.
  Fact 0.1042: Ostby, Miss Helene Ragnhild has 5 in the lifeboat or body column.
  Fact 0.1043: Ovies y Rodriguez, Mr. Servando José Florentino was 36 years old.
  Fact 0.1044: Ovies y Rodriguez, Mr. Servando José Florentino is from Havana, Cuba.
  Fact 0.1045: Ovies y Rodriguez, Mr. Servando José Florentino boarded at Cherbourg.
  Fact 0.1046: Ovies y Rodriguez, Mr. Servando José Florentino's destination is Havana, Cuba.
  Fact 0.1047: Ovies y Rodriguez, Mr. Servando José Florentino has 189 in the lifeboat or body column.
  Fact 0.1048: Parr, Mr. William Henry Marsh was 29 years old.
  Fact 0.1049: Parr, Mr. William Henry Marsh is from Belfast, Ireland, UK.
  Fact 0.1050: Parr, Mr. William Henry Marsh boarded at Belfast.
  Fact 0.1051: Parr, Mr. William Henry Marsh's destination is New York City.
  Fact 0.1052: Partner, Mr. Austin was 40 years old.
  Fact 0.1053: Partner, Mr. Austin is from Tolworth, London, England, UK.
  Fact 0.1054: Partner, Mr. Austin boarded at Southampton.
  Fact 0.1055: Partner, Mr. Austin's destination is Toronto, Ontario, Canada.
  Fact 0.1056: Partner, Mr. Austin has 166 in the lifeboat or body column.
  Fact 0.1057: Pears, Mr. Thomas Clinton was 29 years old.
  Fact 0.1058: Pears, Mr. Thomas Clinton is from Isleworth, London, England, UK.
  Fact 0.1059: Pears, Mr. Thomas Clinton boarded at Southampton.
  Fact 0.1060: Pears, Mr. Thomas Clinton's destination is New York City.
  Fact 0.1061: Pears, Mrs. Edith (née Wearne) was 22 years old.
  Fact 0.1062: Pears, Mrs. Edith (née Wearne) is from Isleworth, London, England, UK.
  Fact 0.1063: Pears, Mrs. Edith (née Wearne) boarded at Southampton.
  Fact 0.1064: Pears, Mrs. Edith (née Wearne)'s destination is New York City.
  Fact 0.1065: Pears, Mrs. Edith (née Wearne) has 8 in the lifeboat or body column.
  Fact 0.1066: Peñasco y Castellana, Mr. Victor was 24 years old.
  Fact 0.1067: Peñasco y Castellana, Mr. Victor is from Madrid, Spain.
  Fact 0.1068: Peñasco y Castellana, Mr. Victor boarded at Cherbourg.
  Fact 0.1069: Peñasco y Castellana, Mr. Victor's destination is New York City.
  Fact 0.1070: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) was 22 years old.
  Fact 0.1071: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) is from Madrid, Spain.
  Fact 0.1072: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) boarded at Cherbourg.
  Fact 0.1073: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo)'s destination is New York City.
  Fact 0.1074: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) has 8 in the lifeboat or body column.
  Fact 0.1075: Doña Fermina Oliva y Ocana was 39 years old.
  Fact 0.1076: Doña Fermina Oliva y Ocana is listed as the maid.
  Fact 0.1077: Doña Fermina Oliva y Ocana is from Madrid, Spain.
  Fact 0.1078: Doña Fermina Oliva y Ocana boarded at Cherbourg.
  Fact 0.1079: Doña Fermina Oliva y Ocana's destination is New York City.
  Fact 0.1080: Doña Fermina Oliva y Ocana has 8 in the lifeboat or body column.
  Fact 0.1081: Peuchen, Major Arthur Godfrey was 52 years old.
  Fact 0.1082: Peuchen, Major Arthur Godfrey is from Toronto, Ontario, Canada.
  Fact 0.1083: Peuchen, Major Arthur Godfrey boarded at Southampton.
  Fact 0.1084: Peuchen, Major Arthur Godfrey's destination is Toronto, Ontario, Canada.
  Fact 0.1085: Peuchen, Major Arthur Godfrey has 6 in the lifeboat or body column.
  Fact 0.1086: Porter, Mr. Walter Chamberlain was 46 years old.
  Fact 0.1087: Porter, Mr. Walter Chamberlain is from Worcester, Massachusetts, US.
  Fact 0.1088: Porter, Mr. Walter Chamberlain boarded at Southampton.
  Fact 0.1089: Porter, Mr. Walter Chamberlain's destination is Worcester, Massachusetts, US.
  Fact 0.1090: Porter, Mr. Walter Chamberlain has 207 in the lifeboat or body column.
  Fact 0.1091: Potter, Mrs. Lily Alexenia (née Wilson) was 56 years old.
  Fact 0.1092: Potter, Mrs. Lily Alexenia (née Wilson) is from Philadelphia, Pennsylvania, US.
  Fact 0.1093: Potter, Mrs. Lily Alexenia (née Wilson) boarded at Cherbourg.
  Fact 0.1094: Potter, Mrs. Lily Alexenia (née Wilson)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.1095: Potter, Mrs. Lily Alexenia (née Wilson) has 7 in the lifeboat or body column.
  Fact 0.1096: Reuchlin, The Honourable Mr. Johan George was 38 years old.
  Fact 0.1097: Reuchlin, The Honourable Mr. Johan George is from Rotterdam, The Netherlands.
  Fact 0.1098: Reuchlin, The Honourable Mr. Johan George boarded at Southampton.
  Fact 0.1099: Reuchlin, The Honourable Mr. Johan George's destination is New York City.
  Fact 0.1100: Rheims, Mr. George Alexander Lucien was 36 years old.
  Fact 0.1101: Rheims, Mr. George Alexander Lucien is from Paris, France.
  Fact 0.1102: Rheims, Mr. George Alexander Lucien boarded at Southampton.
  Fact 0.1103: Rheims, Mr. George Alexander Lucien's destination is New York City.
  Fact 0.1104: Robert, Mrs. Elisabeth Walton (née McMillan) was 43 years old.
  Fact 0.1105: Robert, Mrs. Elisabeth Walton (née McMillan) is from St. Louis, Missouri, US.
  Fact 0.1106: Robert, Mrs. Elisabeth Walton (née McMillan) boarded at Southampton.
  Fact 0.1107: Robert, Mrs. Elisabeth Walton (née McMillan)'s destination is St. Louis, Missouri, US.
  Fact 0.1108: Miss Emilie Kreuchen was 29 years old.
  Fact 0.1109: Miss Emilie Kreuchen is listed as the maid.
  Fact 0.1110: Miss Emilie Kreuchen is from St. Louis, Missouri, US.
  Fact 0.1111: Miss Emilie Kreuchen boarded at Southampton.
  Fact 0.1112: Miss Emilie Kreuchen's destination is St. Louis, Missouri, US.
  Fact 0.1113: Roebling, Mr. Washington Augustus II was 31 years old.
  Fact 0.1114: Roebling, Mr. Washington Augustus II is from Trenton, New Jersey, US.
  Fact 0.1115: Roebling, Mr. Washington Augustus II boarded at Southampton.
  Fact 0.1116: Roebling, Mr. Washington Augustus II's destination is Trenton, New Jersey, US.
  Fact 0.1117: Romaine, Mr. Charles Hallance (alias C. Rolmane) was 45 years old.
  Fact 0.1118: Romaine, Mr. Charles Hallance (alias C. Rolmane) is from New York City.
  Fact 0.1119: Romaine, Mr. Charles Hallance (alias C. Rolmane) boarded at Southampton.
  Fact 0.1120: Romaine, Mr. Charles Hallance (alias C. Rolmane)'s destination is New York City.
  Fact 0.1121: Romaine, Mr. Charles Hallance (alias C. Rolmane) has 9 in the lifeboat or body column.
  Fact 0.1122: Rood, Mr. Hugh Roscoe was 38 years old.
  Fact 0.1123: Rood, Mr. Hugh Roscoe is from Seattle, Washington, US.
  Fact 0.1124: Rood, Mr. Hugh Roscoe boarded at Southampton.
  Fact 0.1125: Rood, Mr. Hugh Roscoe's destination is Seattle, Washington, US.
  Fact 0.1126: Rosenbaum, Miss Edith Louise was 34 years old.
  Fact 0.1127: Rosenbaum, Miss Edith Louise is from Paris, France.
  Fact 0.1128: Rosenbaum, Miss Edith Louise boarded at Cherbourg.
  Fact 0.1129: Rosenbaum, Miss Edith Louise's destination is Cincinnati, Ohio, US.
  Fact 0.1130: Rosenbaum, Miss Edith Louise has 11 in the lifeboat or body column.
  Fact 0.1131: Rosenshine, Mr. George (alias George Thorne) was 46 years old.
  Fact 0.1132: Rosenshine, Mr. George (alias George Thorne) is from New York City.
  Fact 0.1133: Rosenshine, Mr. George (alias George Thorne) boarded at Cherbourg.
  Fact 0.1134: Rosenshine, Mr. George (alias George Thorne)'s destination is New York City.
  Fact 0.1135: Rosenshine, Mr. George (alias George Thorne) has 16 in the lifeboat or body column.
  Fact 0.1136: Ross, Mr. John Hugo was 36 years old.
  Fact 0.1137: Ross, Mr. John Hugo is from Winnipeg, Manitoba, Canada.
  Fact 0.1138: Ross, Mr. John Hugo boarded at Southampton.
  Fact 0.1139: Ross, Mr. John Hugo's destination is Winnipeg, Manitoba, Canada.
  Fact 0.1140: Rothschild, Mr. Martin was 46 years old.
  Fact 0.1141: Rothschild, Mr. Martin is from New York City.
  Fact 0.1142: Rothschild, Mr. Martin boarded at Cherbourg.
  Fact 0.1143: Rothschild, Mr. Martin's destination is New York City.
  Fact 0.1144: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) was 54 years old.
  Fact 0.1145: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) is from New York City.
  Fact 0.1146: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) boarded at Cherbourg.
  Fact 0.1147: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett)'s destination is New York City.
  Fact 0.1148: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) has 6 in the lifeboat or body column.
  Fact 0.1149: Rowe, Mr. Alfred G. was 59 years old.
  Fact 0.1150: Rowe, Mr. Alfred G. is from Liverpool, Merseyside, England, UK.
  Fact 0.1151: Rowe, Mr. Alfred G. boarded at Southampton.
  Fact 0.1152: Rowe, Mr. Alfred G.'s destination is New York City.
  Fact 0.1153: Rowe, Mr. Alfred G. has 109 in the lifeboat or body column.
  Fact 0.1154: Ryerson, Mr. Arthur Larned was 61 years old.
  Fact 0.1155: Ryerson, Mr. Arthur Larned is from Cooperstown, New York, US.
  Fact 0.1156: Ryerson, Mr. Arthur Larned boarded at Cherbourg.
  Fact 0.1157: Ryerson, Mr. Arthur Larned's destination is Cooperstown, New York, US.
  Fact 0.1158: Ryerson, Mrs. Emily Maria (née Borie) was 48 years old.
  Fact 0.1159: Ryerson, Mrs. Emily Maria (née Borie) is from Cooperstown, New York, US.
  Fact 0.1160: Ryerson, Mrs. Emily Maria (née Borie) boarded at Cherbourg.
  Fact 0.1161: Ryerson, Mrs. Emily Maria (née Borie)'s destination is Cooperstown, New York, US.
  Fact 0.1162: Miss Victorine Chaudanson was 36 years old.
  Fact 0.1163: Miss Victorine Chaudanson is listed as the maid.
  Fact 0.1164: Miss Victorine Chaudanson is from Cooperstown, New York, US.
  Fact 0.1165: Miss Victorine Chaudanson boarded at Cherbourg.
  Fact 0.1166: Miss Victorine Chaudanson's destination is Cooperstown, New York, US.
  Fact 0.1167: Ryerson, Miss Susan Parker "Suzette" was 21 years old.
  Fact 0.1168: Ryerson, Miss Susan Parker "Suzette" is from Cooperstown, New York, US.
  Fact 0.1169: Ryerson, Miss Susan Parker "Suzette" boarded at Cherbourg.
  Fact 0.1170: Ryerson, Miss Susan Parker "Suzette"'s destination is Cooperstown, New York, US.
  Fact 0.1171: Ryerson, Miss Emily Borie was 18 years old.
  Fact 0.1172: Ryerson, Miss Emily Borie is from Cooperstown, New York, US.
  Fact 0.1173: Ryerson, Miss Emily Borie boarded at Cherbourg.
  Fact 0.1174: Ryerson, Miss Emily Borie's destination is Cooperstown, New York, US.
  Fact 0.1175: Ryerson, Master John Borie "Jack" was 13 years old.
  Fact 0.1176: Ryerson, Master John Borie "Jack" is from Cooperstown, New York, US.
  Fact 0.1177: Ryerson, Master John Borie "Jack" boarded at Cherbourg.
  Fact 0.1178: Ryerson, Master John Borie "Jack"'s destination is Cooperstown, New York, US.
  Fact 0.1179: Miss Grace Scott Bowen was 45 years old.
  Fact 0.1180: Miss Grace Scott Bowen is listed as the governess.
  Fact 0.1181: Miss Grace Scott Bowen is from Cooperstown, New York, US.
  Fact 0.1182: Miss Grace Scott Bowen boarded at Cherbourg.
  Fact 0.1183: Miss Grace Scott Bowen's destination is Cooperstown, New York, US.
  Fact 0.1184: Saalfeld, Mr. Adolphe was 47 years old.
  Fact 0.1185: Saalfeld, Mr. Adolphe is from Manchester, England, UK.
  Fact 0.1186: Saalfeld, Mr. Adolphe boarded at Southampton.
  Fact 0.1187: Saalfeld, Mr. Adolphe's destination is New York City.
  Fact 0.1188: Salomon, Mr. Abraham Lincoln was 43 years old.
  Fact 0.1189: Salomon, Mr. Abraham Lincoln is from New York City.
  Fact 0.1190: Salomon, Mr. Abraham Lincoln boarded at Cherbourg.
  Fact 0.1191: Salomon, Mr. Abraham Lincoln's destination is New York City.
  Fact 0.1192: Schabert, Mrs. Emma (née Mock) was 35 years old.
  Fact 0.1193: Schabert, Mrs. Emma (née Mock) is from Hamburg, German Empire.
  Fact 0.1194: Schabert, Mrs. Emma (née Mock) boarded at Southampton.
  Fact 0.1195: Schabert, Mrs. Emma (née Mock)'s destination is New York City.
  Fact 0.1196: Schabert, Mrs. Emma (née Mock) has 11 in the lifeboat or body column.
  Fact 0.1197: Seward, Mr. Frederic Kimber was 34 years old.
  Fact 0.1198: Seward, Mr. Frederic Kimber is from New York City.
  Fact 0.1199: Seward, Mr. Frederic Kimber boarded at Southampton.
  Fact 0.1200: Seward, Mr. Frederic Kimber's destination is New York City.
  Fact 0.1201: Seward, Mr. Frederic Kimber has 7 in the lifeboat or body column.
  Fact 0.1202: Silverthorne, Mr. Spencer Victor was 35 years old.
  Fact 0.1203: Silverthorne, Mr. Spencer Victor is from St. Louis, Missouri, US.
  Fact 0.1204: Silverthorne, Mr. Spencer Victor boarded at Southampton.
  Fact 0.1205: Silverthorne, Mr. Spencer Victor's destination is St. Louis, Missouri, US.
  Fact 0.1206: Silverthorne, Mr. Spencer Victor has 5 in the lifeboat or body column.
  Fact 0.1207: Silvey, Mr. William Baird was 50 years old.
  Fact 0.1208: Silvey, Mr. William Baird is from Duluth, Minnesota, US.
  Fact 0.1209: Silvey, Mr. William Baird boarded at Cherbourg.
  Fact 0.1210: Silvey, Mr. William Baird's destination is Duluth, Minnesota, US.
  Fact 0.1211: Silvey, Mrs. Alice (née Munger) was 39 years old.
  Fact 0.1212: Silvey, Mrs. Alice (née Munger) is from Duluth, Minnesota, US.
  Fact 0.1213: Silvey, Mrs. Alice (née Munger) boarded at Cherbourg.
  Fact 0.1214: Silvey, Mrs. Alice (née Munger)'s destination is Duluth, Minnesota, US.
  Fact 0.1215: Silvey, Mrs. Alice (née Munger) has 11 in the lifeboat or body column.
  Fact 0.1216: Simonius-Blumer, Colonel Alfons was 56 years old.
  Fact 0.1217: Simonius-Blumer, Colonel Alfons is from Basel, Switzerland.
  Fact 0.1218: Simonius-Blumer, Colonel Alfons boarded at Southampton.
  Fact 0.1219: Simonius-Blumer, Colonel Alfons's destination is New York City.
  Fact 0.1220: Sloper, Mr. William Thompson was 28 years old.
  Fact 0.1221: Sloper, Mr. William Thompson is from New Britain, Connecticut, US.
  Fact 0.1222: Sloper, Mr. William Thompson boarded at Southampton.
  Fact 0.1223: Sloper, Mr. William Thompson's destination is New Britain, Connecticut, US.
  Fact 0.1224: Sloper, Mr. William Thompson has 7 in the lifeboat or body column.
  Fact 0.1225: Smart, Mr. John Montgomery was 56 years old.
  Fact 0.1226: Smart, Mr. John Montgomery is from Kildale, North Yorkshire, England, UK.
  Fact 0.1227: Smart, Mr. John Montgomery boarded at Southampton.
  Fact 0.1228: Smart, Mr. John Montgomery's destination is New York City.
  Fact 0.1229: Smith, Mr. James Clinch was 56 years old.
  Fact 0.1230: Smith, Mr. James Clinch is from Paris, France.
  Fact 0.1231: Smith, Mr. James Clinch boarded at Cherbourg.
  Fact 0.1232: Smith, Mr. James Clinch's destination is Long Island, New York, US.
  Fact 0.1233: Smith, Mr. Lucien Philip was 24 years old.
  Fact 0.1234: Smith, Mr. Lucien Philip is from Huntington, West Virginia, US.
  Fact 0.1235: Smith, Mr. Lucien Philip boarded at Cherbourg.
  Fact 0.1236: Smith, Mr. Lucien Philip's destination is Huntington, West Virginia, US.
  Fact 0.1237: Smith, Mrs. Mary Eloise (née Hughes) was 18 years old.
  Fact 0.1238: Smith, Mrs. Mary Eloise (née Hughes) is from Huntington, West Virginia, US.
  Fact 0.1239: Smith, Mrs. Mary Eloise (née Hughes) boarded at Cherbourg.
  Fact 0.1240: Smith, Mrs. Mary Eloise (née Hughes)'s destination is Huntington, West Virginia, US.
  Fact 0.1241: Smith, Mrs. Mary Eloise (née Hughes) has 6 in the lifeboat or body column.
  Fact 0.1242: Smith, Mr. Richard William was 57 years old.
  Fact 0.1243: Smith, Mr. Richard William is from Streatham, London, England, UK.
  Fact 0.1244: Smith, Mr. Richard William boarded at Southampton.
  Fact 0.1245: Smith, Mr. Richard William's destination is New York City.
  Fact 0.1246: Snyder, Mr. John Pillsbury was 24 years old.
  Fact 0.1247: Snyder, Mr. John Pillsbury is from Minneapolis, Minnesota, US.
  Fact 0.1248: Snyder, Mr. John Pillsbury boarded at Southampton.
  Fact 0.1249: Snyder, Mr. John Pillsbury's destination is Minneapolis, Minnesota, US.
  Fact 0.1250: Snyder, Mr. John Pillsbury has 7 in the lifeboat or body column.
  Fact 0.1251: Snyder, Mrs. Nellie (née Stevenson) was 23 years old.
  Fact 0.1252: Snyder, Mrs. Nellie (née Stevenson) is from Minneapolis, Minnesota, US.
  Fact 0.1253: Snyder, Mrs. Nellie (née Stevenson) boarded at Southampton.
  Fact 0.1254: Snyder, Mrs. Nellie (née Stevenson)'s destination is Minneapolis, Minnesota, US.
  Fact 0.1255: Snyder, Mrs. Nellie (née Stevenson) has 7 in the lifeboat or body column.
  Fact 0.1256: Spedden, Mr. Frederic Oakley was 45 years old.
  Fact 0.1257: Spedden, Mr. Frederic Oakley is from Tuxedo Park, New York, US.
  Fact 0.1258: Spedden, Mr. Frederic Oakley boarded at Cherbourg.
  Fact 0.1259: Spedden, Mr. Frederic Oakley's destination is Tuxedo Park, New York, US.
  Fact 0.1260: Spedden, Mrs. Margaretta Corning (née Stone) was 39 years old.
  Fact 0.1261: Spedden, Mrs. Margaretta Corning (née Stone) is from Tuxedo Park, New York, US.
  Fact 0.1262: Spedden, Mrs. Margaretta Corning (née Stone) boarded at Cherbourg.
  Fact 0.1263: Spedden, Mrs. Margaretta Corning (née Stone)'s destination is Tuxedo Park, New York, US.
  Fact 0.1264: Miss Helen Alice Wilson was 31 years old.
  Fact 0.1265: Miss Helen Alice Wilson is listed as the maid.
  Fact 0.1266: Miss Helen Alice Wilson is from Tuxedo Park, New York, US.
  Fact 0.1267: Miss Helen Alice Wilson boarded at Cherbourg.
  Fact 0.1268: Miss Helen Alice Wilson's destination is Tuxedo Park, New York, US.
  Fact 0.1269: Spedden, Master Robert Douglas was 6 years old.
  Fact 0.1270: Spedden, Master Robert Douglas is from Tuxedo Park, New York, US.
  Fact 0.1271: Spedden, Master Robert Douglas boarded at Cherbourg.
  Fact 0.1272: Spedden, Master Robert Douglas's destination is Tuxedo Park, New York, US.
  Fact 0.1273: Miss Elizabeth Margaret Burns was 41 years old.
  Fact 0.1274: Miss Elizabeth Margaret Burns is listed as the nurse.
  Fact 0.1275: Miss Elizabeth Margaret Burns is from Tuxedo Park, New York, US.
  Fact 0.1276: Miss Elizabeth Margaret Burns boarded at Cherbourg.
  Fact 0.1277: Miss Elizabeth Margaret Burns's destination is Tuxedo Park, New York, US.
  Fact 0.1278: Spencer, Mr. William Augustus was 57 years old.
  Fact 0.1279: Spencer, Mr. William Augustus is from New York City.
  Fact 0.1280: Spencer, Mr. William Augustus boarded at Cherbourg.
  Fact 0.1281: Spencer, Mr. William Augustus's destination is New York City.
  Fact 0.1282: Spencer, Mrs. Marie Eugénie (née Demougeot) was 45 years old.
  Fact 0.1283: Spencer, Mrs. Marie Eugénie (née Demougeot) is from New York City.
  Fact 0.1284: Spencer, Mrs. Marie Eugénie (née Demougeot) boarded at Cherbourg.
  Fact 0.1285: Spencer, Mrs. Marie Eugénie (née Demougeot)'s destination is New York City.
  Fact 0.1286: Spencer, Mrs. Marie Eugénie (née Demougeot) has 6 in the lifeboat or body column.
  Fact 0.1287: Miss Eugenie Elise Lurette was 59 years old.
  Fact 0.1288: Miss Eugenie Elise Lurette is listed as the maid.
  Fact 0.1289: Miss Eugenie Elise Lurette is from New York City.
  Fact 0.1290: Miss Eugenie Elise Lurette boarded at Cherbourg.
  Fact 0.1291: Miss Eugenie Elise Lurette's destination is New York City.
  Fact 0.1292: Miss Eugenie Elise Lurette has 6 in the lifeboat or body column.
  Fact 0.1293: Stähelin-Maeglin, Dr. Max was 32 years old.
  Fact 0.1294: Stähelin-Maeglin, Dr. Max is from Basel, Switzerland.
  Fact 0.1295: Stähelin-Maeglin, Dr. Max boarded at Southampton.
  Fact 0.1296: Stähelin-Maeglin, Dr. Max's destination is New York City.
  Fact 0.1297: Stead, Mr. William Thomas was 62 years old.
  Fact 0.1298: Stead, Mr. William Thomas is from London, England, UK.
  Fact 0.1299: Stead, Mr. William Thomas boarded at Southampton.
  Fact 0.1300: Stead, Mr. William Thomas's destination is New York City.
  Fact 0.1301: Stengel, Mr. Charles Emil Henry was 54 years old.
  Fact 0.1302: Stengel, Mr. Charles Emil Henry is from Newark, New Jersey, US.
  Fact 0.1303: Stengel, Mr. Charles Emil Henry boarded at Cherbourg.
  Fact 0.1304: Stengel, Mr. Charles Emil Henry's destination is Newark, New Jersey, US.
  Fact 0.1305: Stengel, Mrs. Annie May (née Morris) was 44 years old.
  Fact 0.1306: Stengel, Mrs. Annie May (née Morris) is from Newark, New Jersey, US.
  Fact 0.1307: Stengel, Mrs. Annie May (née Morris) boarded at Cherbourg.
  Fact 0.1308: Stengel, Mrs. Annie May (née Morris)'s destination is Newark, New Jersey, US.
  Fact 0.1309: Stengel, Mrs. Annie May (née Morris) has 5 in the lifeboat or body column.
  Fact 0.1310: Stephenson, Mrs. Martha (née Eustis) was 52 years old.
  Fact 0.1311: Stephenson, Mrs. Martha (née Eustis) is from Paris, France.
  Fact 0.1312: Stephenson, Mrs. Martha (née Eustis) boarded at Cherbourg.
  Fact 0.1313: Stephenson, Mrs. Martha (née Eustis)'s destination is Haverford, Pennsylvania, US.
  Fact 0.1314: Stewart, Mr. Albert A. was 54 years old.
  Fact 0.1315: Stewart, Mr. Albert A. is from Cincinnati, Ohio, US.
  Fact 0.1316: Stewart, Mr. Albert A. boarded at Cherbourg.
  Fact 0.1317: Stewart, Mr. Albert A.'s destination is Gallipolis, Ohio, US.
  Fact 0.1318: Stone, Mrs. Martha Evelyn (née Stevens) was 62 years old.
  Fact 0.1319: Stone, Mrs. Martha Evelyn (née Stevens) is from New York City.
  Fact 0.1320: Stone, Mrs. Martha Evelyn (née Stevens) boarded at Southampton.
  Fact 0.1321: Stone, Mrs. Martha Evelyn (née Stevens)'s destination is New York City.
  Fact 0.1322: Stone, Mrs. Martha Evelyn (née Stevens) has 6 in the lifeboat or body column.
  Fact 0.1323: Miss Amelie "Amelia" Icard was 38 years old.
  Fact 0.1324: Miss Amelie "Amelia" Icard is listed as the maid.
  Fact 0.1325: Miss Amelie "Amelia" Icard is from New York City.
  Fact 0.1326: Miss Amelie "Amelia" Icard boarded at Southampton.
  Fact 0.1327: Miss Amelie "Amelia" Icard's destination is New York City.
  Fact 0.1328: Miss Amelie "Amelia" Icard has 6 in the lifeboat or body column.
  Fact 0.1329: Straus, Mr. Isidor was 67 years old.
  Fact 0.1330: Straus, Mr. Isidor is from New York City.
  Fact 0.1331: Straus, Mr. Isidor boarded at Southampton.
  Fact 0.1332: Straus, Mr. Isidor's destination is New York City.
  Fact 0.1333: Straus, Mr. Isidor has 96 in the lifeboat or body column.
  Fact 0.1334: Mr. John Farthing was 57 years old.
  Fact 0.1335: Mr. John Farthing is listed as the valet.
  Fact 0.1336: Mr. John Farthing is from New York City.
  Fact 0.1337: Mr. John Farthing boarded at Southampton.
  Fact 0.1338: Mr. John Farthing's destination is New York City.
  Fact 0.1339: Straus, Mrs. Rosalie Ida (née Blun) was 63 years old.
  Fact 0.1340: Straus, Mrs. Rosalie Ida (née Blun) is from New York City.
  Fact 0.1341: Straus, Mrs. Rosalie Ida (née Blun) boarded at Southampton.
  Fact 0.1342: Straus, Mrs. Rosalie Ida (née Blun)'s destination is New York City.
  Fact 0.1343: Miss Ellen Bird was 31 years old.
  Fact 0.1344: Miss Ellen Bird is listed as the maid.
  Fact 0.1345: Miss Ellen Bird is from New York City.
  Fact 0.1346: Miss Ellen Bird boarded at Southampton.
  Fact 0.1347: Miss Ellen Bird's destination is New York City.
  Fact 0.1348: Miss Ellen Bird has 8 in the lifeboat or body column.
  Fact 0.1349: Sutton, Mr. Frederick was 61 years old.
  Fact 0.1350: Sutton, Mr. Frederick is from Haddonfield, New Jersey, US.
  Fact 0.1351: Sutton, Mr. Frederick boarded at Southampton.
  Fact 0.1352: Sutton, Mr. Frederick's destination is Haddonfield, New Jersey, US.
  Fact 0.1353: Sutton, Mr. Frederick has 46 in the lifeboat or body column.
  Fact 0.1354: Swift, Mrs. Margaret Welles (née Barron) was 46 years old.
  Fact 0.1355: Swift, Mrs. Margaret Welles (née Barron) is from New York City.
  Fact 0.1356: Swift, Mrs. Margaret Welles (née Barron) boarded at Southampton.
  Fact 0.1357: Swift, Mrs. Margaret Welles (née Barron)'s destination is New York City.
  Fact 0.1358: Swift, Mrs. Margaret Welles (née Barron) has 8 in the lifeboat or body column.
  Fact 0.1359: Taussig, Mr. Emil was 52 years old.
  Fact 0.1360: Taussig, Mr. Emil is from New York City.
  Fact 0.1361: Taussig, Mr. Emil boarded at Southampton.
  Fact 0.1362: Taussig, Mr. Emil's destination is New York City.
  Fact 0.1363: Taussig, Mrs. Tillie (née Mandelbaum) was 39 years old.
  Fact 0.1364: Taussig, Mrs. Tillie (née Mandelbaum) is from New York City.
  Fact 0.1365: Taussig, Mrs. Tillie (née Mandelbaum) boarded at Southampton.
  Fact 0.1366: Taussig, Mrs. Tillie (née Mandelbaum)'s destination is New York City.
  Fact 0.1367: Taussig, Mrs. Tillie (née Mandelbaum) has 8 in the lifeboat or body column.
  Fact 0.1368: Taussig, Miss Ruth was 18 years old.
  Fact 0.1369: Taussig, Miss Ruth is from New York City.
  Fact 0.1370: Taussig, Miss Ruth boarded at Southampton.
  Fact 0.1371: Taussig, Miss Ruth's destination is New York City.
  Fact 0.1372: Taussig, Miss Ruth has 8 in the lifeboat or body column.
  Fact 0.1373: Taylor, Mr. Elmer Zebley was 48 years old.
  Fact 0.1374: Taylor, Mr. Elmer Zebley is from London, England, UK.
  Fact 0.1375: Taylor, Mr. Elmer Zebley boarded at Southampton.
  Fact 0.1376: Taylor, Mr. Elmer Zebley's destination is East Orange, New Jersey, US.
  Fact 0.1377: Taylor, Mr. Elmer Zebley has 5 in the lifeboat or body column.
  Fact 0.1378: Taylor, Mrs. Juliet Cummins (née Wright) was 48 years old.
  Fact 0.1379: Taylor, Mrs. Juliet Cummins (née Wright) is from London, England, UK.
  Fact 0.1380: Taylor, Mrs. Juliet Cummins (née Wright) boarded at Southampton.
  Fact 0.1381: Taylor, Mrs. Juliet Cummins (née Wright)'s destination is East Orange, New Jersey, US.
  Fact 0.1382: Taylor, Mrs. Juliet Cummins (née Wright) has 5 in the lifeboat or body column.
  Fact 0.1383: Thayer, Mr. John Borland II was 49 years old.
  Fact 0.1384: Thayer, Mr. John Borland II is from Haverford, Pennsylvania, US.
  Fact 0.1385: Thayer, Mr. John Borland II boarded at Cherbourg.
  Fact 0.1386: Thayer, Mr. John Borland II's destination is Haverford, Pennsylvania, US.
  Fact 0.1387: Thayer, Mrs. Marian Longsteth (née Morris) was 39 years old.
  Fact 0.1388: Thayer, Mrs. Marian Longsteth (née Morris) is from Haverford, Pennsylvania, US.
  Fact 0.1389: Thayer, Mrs. Marian Longsteth (née Morris) boarded at Cherbourg.
  Fact 0.1390: Thayer, Mrs. Marian Longsteth (née Morris)'s destination is Haverford, Pennsylvania, US.
  Fact 0.1391: Miss Margaret Fleming was 42 years old.
  Fact 0.1392: Miss Margaret Fleming is listed as the maid.
  Fact 0.1393: Miss Margaret Fleming is from Haverford, Pennsylvania, US.
  Fact 0.1394: Miss Margaret Fleming boarded at Cherbourg.
  Fact 0.1395: Miss Margaret Fleming's destination is Haverford, Pennsylvania, US.
  Fact 0.1396: Thayer, Mr. John Borland "Jack" III was 17 years old.
  Fact 0.1397: Thayer, Mr. John Borland "Jack" III is from Haverford, Pennsylvania, US.
  Fact 0.1398: Thayer, Mr. John Borland "Jack" III boarded at Cherbourg.
  Fact 0.1399: Thayer, Mr. John Borland "Jack" III's destination is Haverford, Pennsylvania, US.
  Fact 0.1400: Thorne, Mrs. Gertrude Maybelle (née McMinn) was 37 years old.
  Fact 0.1401: Thorne, Mrs. Gertrude Maybelle (née McMinn) is from New York City, New York, US.
  Fact 0.1402: Thorne, Mrs. Gertrude Maybelle (née McMinn) boarded at Cherbourg.
  Fact 0.1403: Thorne, Mrs. Gertrude Maybelle (née McMinn)'s destination is New York City, New York, US.
  Fact 0.1404: Tucker, Mr. Gilbert Milligan Jr. was 31 years old.
  Fact 0.1405: Tucker, Mr. Gilbert Milligan Jr. is from Albany, New York, US.
  Fact 0.1406: Tucker, Mr. Gilbert Milligan Jr. boarded at Cherbourg.
  Fact 0.1407: Tucker, Mr. Gilbert Milligan Jr.'s destination is Albany, New York, US.
  Fact 0.1408: Tucker, Mr. Gilbert Milligan Jr. has 7 in the lifeboat or body column.
  Fact 0.1409: Uruchurtu, Don Manuel E. was 40 years old.
  Fact 0.1410: Uruchurtu, Don Manuel E. is from Mexico City, Mexico.
  Fact 0.1411: Uruchurtu, Don Manuel E. boarded at Cherbourg.
  Fact 0.1412: Uruchurtu, Don Manuel E.'s destination is Mexico City, Mexico.
  Fact 0.1413: Van der Hoef, Mr. Wyckoff was 61 years old.
  Fact 0.1414: Van der Hoef, Mr. Wyckoff is from Brooklyn, New York, US.
  Fact 0.1415: Van der Hoef, Mr. Wyckoff boarded at Belfast.
  Fact 0.1416: Van der Hoef, Mr. Wyckoff's destination is Brooklyn, New York, US.
  Fact 0.1417: Van der Hoef, Mr. Wyckoff has 245 in the lifeboat or body column.
  Fact 0.1418: Walker, Mr. William Anderson was 48 years old.
  Fact 0.1419: Walker, Mr. William Anderson is from East Orange, New Jersey, US.
  Fact 0.1420: Walker, Mr. William Anderson boarded at Southampton.
  Fact 0.1421: Walker, Mr. William Anderson's destination is East Orange, New Jersey, US.
  Fact 0.1422: Warren, Mr. Frank Manley was 63 years old.
  Fact 0.1423: Warren, Mr. Frank Manley is from Portland, Oregon, US.
  Fact 0.1424: Warren, Mr. Frank Manley boarded at Cherbourg.
  Fact 0.1425: Warren, Mr. Frank Manley's destination is Portland, Oregon, US.
  Fact 0.1426: Warren, Mrs. Anna Sophia (née Atkinson) was 60 years old.
  Fact 0.1427: Warren, Mrs. Anna Sophia (née Atkinson) is from Portland, Oregon, US.
  Fact 0.1428: Warren, Mrs. Anna Sophia (née Atkinson) boarded at Cherbourg.
  Fact 0.1429: Warren, Mrs. Anna Sophia (née Atkinson)'s destination is Portland, Oregon, US.
  Fact 0.1430: Warren, Mrs. Anna Sophia (née Atkinson) has 5 in the lifeboat or body column.
  Fact 0.1431: Weir, Colonel John was 59 years old.
  Fact 0.1432: Weir, Colonel John is from New York City.
  Fact 0.1433: Weir, Colonel John boarded at Southampton.
  Fact 0.1434: Weir, Colonel John's destination is New York City.
  Fact 0.1435: White, Mr. Percival Wayland was 54 years old.
  Fact 0.1436: White, Mr. Percival Wayland is from Brunswick, Maine, US.
  Fact 0.1437: White, Mr. Percival Wayland boarded at Southampton.
  Fact 0.1438: White, Mr. Percival Wayland's destination is Brunswick, Maine, US.
  Fact 0.1439: White, Mr. Richard Frasar was 21 years old.
  Fact 0.1440: White, Mr. Richard Frasar is from Brunswick, Maine, US.
  Fact 0.1441: White, Mr. Richard Frasar boarded at Southampton.
  Fact 0.1442: White, Mr. Richard Frasar's destination is Brunswick, Maine, US.
  Fact 0.1443: White, Mr. Richard Frasar has 169 in the lifeboat or body column.
  Fact 0.1444: White, Mrs. Ella (née Holmes) was 55 years old.
  Fact 0.1445: White, Mrs. Ella (née Holmes) is from New York City.
  Fact 0.1446: White, Mrs. Ella (née Holmes) boarded at Cherbourg.
  Fact 0.1447: White, Mrs. Ella (née Holmes)'s destination is New York City.
  Fact 0.1448: White, Mrs. Ella (née Holmes) has 8 in the lifeboat or body column.
  Fact 0.1449: Miss Amelia Mayo "Nellie" Bessette was 39 years old.
  Fact 0.1450: Miss Amelia Mayo "Nellie" Bessette is listed as the maid.
  Fact 0.1451: Miss Amelia Mayo "Nellie" Bessette is from New York City.
  Fact 0.1452: Miss Amelia Mayo "Nellie" Bessette boarded at Cherbourg.
  Fact 0.1453: Miss Amelia Mayo "Nellie" Bessette's destination is New York City.
  Fact 0.1454: Miss Amelia Mayo "Nellie" Bessette has 8 in the lifeboat or body column.
  Fact 0.1455: Mr. Sante Righini was 22 years old.
  Fact 0.1456: Mr. Sante Righini is listed as the manservant.
  Fact 0.1457: Mr. Sante Righini is from New York City.
  Fact 0.1458: Mr. Sante Righini boarded at Cherbourg.
  Fact 0.1459: Mr. Sante Righini's destination is New York City.
  Fact 0.1460: Mr. Sante Righini has 232 in the lifeboat or body column.
  Fact 0.1461: Wick, Colonel George Dennick was 58 years old.
  Fact 0.1462: Wick, Colonel George Dennick is from Youngstown, Ohio, US.
  Fact 0.1463: Wick, Colonel George Dennick boarded at Southampton.
  Fact 0.1464: Wick, Colonel George Dennick's destination is Youngstown, Ohio, US.
  Fact 0.1465: Wick, Colonel George Dennick has 8 in the lifeboat or body column.
  Fact 0.1466: Wick, Mrs. Mary (née Hitchcock) was 45 years old.
  Fact 0.1467: Wick, Mrs. Mary (née Hitchcock) is from Youngstown, Ohio, US.
  Fact 0.1468: Wick, Mrs. Mary (née Hitchcock) boarded at Southampton.
  Fact 0.1469: Wick, Mrs. Mary (née Hitchcock)'s destination is Youngstown, Ohio, US.
  Fact 0.1470: Wick, Mrs. Mary (née Hitchcock) has 8 in the lifeboat or body column.
  Fact 0.1471: Wick, Miss Mary Natalie was 31 years old.
  Fact 0.1472: Wick, Miss Mary Natalie is from Youngstown, Ohio, US.
  Fact 0.1473: Wick, Miss Mary Natalie boarded at Southampton.
  Fact 0.1474: Wick, Miss Mary Natalie's destination is Youngstown, Ohio, US.
  Fact 0.1475: Wick, Miss Mary Natalie has 8 in the lifeboat or body column.
  Fact 0.1476: Widener, Mr. George Dunton was 50 years old.
  Fact 0.1477: Widener, Mr. George Dunton is from Philadelphia, Pennsylvania, US.
  Fact 0.1478: Widener, Mr. George Dunton boarded at Southampton.
  Fact 0.1479: Widener, Mr. George Dunton's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1480: Mr. Edwin Herbert Keeping was 33 years old.
  Fact 0.1481: Mr. Edwin Herbert Keeping is listed as the valet.
  Fact 0.1482: Mr. Edwin Herbert Keeping is from Philadelphia, Pennsylvania, US.
  Fact 0.1483: Mr. Edwin Herbert Keeping boarded at Southampton.
  Fact 0.1484: Mr. Edwin Herbert Keeping's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1485: Mr. Edwin Herbert Keeping has 45 in the lifeboat or body column.
  Fact 0.1486: Widener, Mrs. Eleanor (née Elkins) was 50 years old.
  Fact 0.1487: Widener, Mrs. Eleanor (née Elkins) is from Philadelphia, Pennsylvania, US.
  Fact 0.1488: Widener, Mrs. Eleanor (née Elkins) boarded at Southampton.
  Fact 0.1489: Widener, Mrs. Eleanor (née Elkins)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.1490: Miss Amalie Henriette "Emily" Gieger was 35 years old.
  Fact 0.1491: Miss Amalie Henriette "Emily" Gieger is listed as the maid.
  Fact 0.1492: Miss Amalie Henriette "Emily" Gieger is from Philadelphia, Pennsylvania, US.
  Fact 0.1493: Miss Amalie Henriette "Emily" Gieger boarded at Southampton.
  Fact 0.1494: Miss Amalie Henriette "Emily" Gieger's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1495: Widener, Mr. Harry Elkins was 27 years old.
  Fact 0.1496: Widener, Mr. Harry Elkins is from Philadelphia, Pennsylvania, US.
  Fact 0.1497: Widener, Mr. Harry Elkins boarded at Southampton.
  Fact 0.1498: Widener, Mr. Harry Elkins's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1499: Willard, Miss Constance was 21 years old.
  Fact 0.1500: Willard, Miss Constance is from Duluth, Minnesota, US.
  Fact 0.1501: Willard, Miss Constance boarded at Southampton.
  Fact 0.1502: Willard, Miss Constance's destination is Duluth, Minnesota, US.
  Fact 0.1503: Willard, Miss Constance has 8 in the lifeboat or body column.
  Fact 0.1504: Williams, Mr. Charles Duane was 51 years old.
  Fact 0.1505: Williams, Mr. Charles Duane is from Geneva, Switzerland.
  Fact 0.1506: Williams, Mr. Charles Duane boarded at Cherbourg.
  Fact 0.1507: Williams, Mr. Charles Duane's destination is Radnor, Pennsylvania, US.
  Fact 0.1508: Williams, Mr. Richard Norris II was 21 years old.
  Fact 0.1509: Williams, Mr. Richard Norris II is from Geneva, Switzerland.
  Fact 0.1510: Williams, Mr. Richard Norris II boarded at Cherbourg.
  Fact 0.1511: Williams, Mr. Richard Norris II's destination is Radnor, Pennsylvania, US.
  Fact 0.1512: Woolner, Mr. Hugh was 45 years old.
  Fact 0.1513: Woolner, Mr. Hugh is from London, England, UK.
  Fact 0.1514: Woolner, Mr. Hugh boarded at Southampton.
  Fact 0.1515: Woolner, Mr. Hugh's destination is New York City.
  Fact 0.1516: Wright, Mr. George was 62 years old.
  Fact 0.1517: Wright, Mr. George is from Halifax, Nova Scotia, Canada.
  Fact 0.1518: Wright, Mr. George boarded at Southampton.
  Fact 0.1519: Wright, Mr. George's destination is Halifax, Nova Scotia, Canada.
  Fact 0.1520: Young, Miss Marie Grice was 36 years old.
  Fact 0.1521: Young, Miss Marie Grice is from New York City.
  Fact 0.1522: Young, Miss Marie Grice boarded at Cherbourg.
  Fact 0.1523: Young, Miss Marie Grice's destination is New York City.
  Fact 0.1524: Young, Miss Marie Grice has 8 in the lifeboat or body column.

Retrieved items:
<item 1>
Graph query row:
survivor: J. Bruce Ismay
relationship: RELATED_TO
connected_entity: RMS Titanic
</item 1>

<item 2>
Passengers of the RMS Titanic
( Halifax ) Related Ships RMS Olympic RMS Carpathia HMHS Britannic CS Mackay - Bennett SS Californian Replica Titanic Titanic II Others White Star Line Titanic
Historical Society Encyclopedia Titanica Halomonas titanicae Women and children first SOS Robert Ballard Last remaining survivors of the RMS Titanic disaster
2000 -- 2009 Millvina Dean ( 1912 -- 2009 ) Barbara West ( 1911 -- 2007 ) Lillian Asplund ( 1906 -- 2006 ) Winnifred Quick ( 1904 -- 2002 ) Michel Navratil (
1908 -- 2001 ) 1995 -- 1999 Eleanor Johnson ( 1910 -- 1998 ) Louise Laroche ( 1910 -- 1998 ) Edith Brown ( 1896 -- 1997 ) Eva Hart ( 1905 -- 1996 ) Beatrice
Sandström ( 1910 -- 1995 ) 1990 -- 1994 Robertha `` Bertha '' Watt ( 1899 -- 1993 ) Ellen Shine ( 1891 -- 1993 ) Marjorie Newell ( 1889 -- 1992 ) Louise
</item 2>

<item 3>
Passengers of the RMS Titanic
Victims whose remains were recovered after the sinking are listed with a superscript next to the body number , indicating the recovery vessel : MB -- CS Mackay
- Bennett ( bodies 1 -- 306 ) M -- CS Minia ( bodies 307 -- 323 ) MM -- CGS Montmagny ( bodies 326 -- 329 ) A -- SS Algerine ( body 330 ) O -- RMS Oceanic (
bodies 331 -- 333 ) I -- SS Ilford ( body 334 ) OT -- SS Ottawa ( body 335 ) Numbers 324 and 325 were unused , and the six bodies buried at sea by the Carpathia
also went unnumbered . First class ( edit ) Name Age Hometown Boarded Destination Lifeboat Body Allen , Miss Elizabeth Walton 29 St. Louis , Missouri , US
Southampton St. Louis , Missouri , US Allison , Mr. Hudson Joshua Creighton 30 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 135 and
chauffeur , Mr.
</item 3>

<item 4>
Passengers of the RMS Titanic
ISBN 978 - 0 - 14 - 004757 - 8 . Jump up ^ `` Passenger List and Survivors of Steamship Titanic '' . United States Senate Inquiry. 30 July 1912 . Retrieved 15
July 2011 . Jump up ^ Hall , Wayne ( 1986 ) . `` Social Class and Survival on the SS Titanic '' ( PDF ) . Social Science & Medicine . 22 ( 6 ) : 687 -- 690 .
Jump up ^ Barratt , Nick ( 2009 ) . Lost Voices From the Titanic : The Definitive Oral History . London : Random House . p. 93 . ISBN 978 - 1 - 84809 - 151 - 1
. Jump up ^ Howells , Richard ( 1999 ) . The Myth of the Titanic . United Kingdom : MacMillan Press . p. 18 . ISBN 978 - 0 - 333 - 72597 - 9 .
</item 4>

<item 5>
Passengers of the RMS Titanic
Collapsible B floated away from the ship upside down , while Collapsible A became half - filled with water after the supports for its canvas sides were broken
in the fall from the roof of the officers ' quarters . There were arguments in some of the lifeboats about going back to pick up people in the water , but many
survivors were afraid of being swamped by people trying to climb into the lifeboat or being pulled down by the suction from the sinking Titanic , though it
turned out that there had been very little suction . At 2 : 20 am , Titanic herself sank . A small number of passengers and crew were able to make their way to
the two unlaunched collapsible boats , surviving for several hours ( some still clinging to the overturned Collapsible B) until they were rescued by Fifth
Officer Harold Lowe . The RMS Carpathia arriving at Pier 54 with Titanic survivors . At 4 : 10 am , the RMS Carpathia arrived at the site of the sinking and
began rescuing survivors . By 8 : 30 am , she picked up the last lifeboat with survivors and left the area at 08 : 50 bound for Pier 54 in New York City .
</item 5>

<item 6>
Passengers of the RMS Titanic
Although several people managed to reach this lifeboat , three died during the night . When Fifth Officer Harold Lowe and six crewmen returned to the wreck site
after the sinking with an empty lifeboat to pick up survivors , they rescued surviving passengers from Collapsible A , but left the three dead bodies in the
boat : Thomson Beattie , a first - class passenger , and two crew members , a fireman and a seaman . After their retrieval from Collapsible A by Oceanic , the
bodies were buried at sea . Passenger list ( edit ) Breakdown of casualties according to the British Board of Trade report ( click for detail ) The following is
a full list of known passengers who sailed on the maiden voyage of the RMS Titanic . Included in this list are the nine - member Guarantee Group and the eight
members of the ship 's band , listed as both passengers and crew . They are also included in the list of crew members on board RMS Titanic . Passengers are
colour - coded , indicating whether they were saved or perished . The passenger did not survive The passenger survived Survivors are listed with the lifeboat
from which they were known to be rescued .
</item 6>

<item 7>
Passengers of the RMS Titanic
Frank M. Warren , Sr . George Dennick Wick George Dunton Widener Harry Elkins Widener Duane Williams George Henry Wright Roderick Chisholm Survivors ( last
living ) Rhoda Abbott Trevor Allison Lillian Asplund Madeleine Astor Ruth Becker Lawrence Beesley Karl Behr Dickinson Bishop Mauritz Håkan Björnström -
Steffansson Elsie Bowerman Francis Browne Margaret `` Molly '' Brown Daniel Buckley Alden Caldwell Helen Churchill Candee Charlotte Drake Cardeza Lucile Carter
Gladys Cherry Millvina Dean Margaret Devaney Sir Cosmo Duff - Gordon Lucy , Lady Duff - Gordon Ethel Flora Fortune Dorothy Gibson Archibald Gracie IV Frank John
William Goldsmith Edith Haisman Henry S. Harper Eva Hart Margaret Bechstein Hays Masabumi Hosono J. Bruce Ismay Eleanor Ileen Johnson Louise Laroche Louise Kink
Margaret Mannion Michel Marcel Navratil Alfred Nourney Arthur Godfrey Peuchen Winnifred Quick Marjorie Newell Robb Edith Rosenbaum Noël Leslie ,
</item 7>

<item 8>
Passengers of the RMS Titanic
Of the 711 passengers and crew rescued by the Carpathia , six , including first class passenger William F. Hoyt , either died in a lifeboat during the night or
on board the Carpathia the next morning , and were buried at sea . In the days following the sinking , several ships sailed to the disaster area to recover
victims ' bodies . The White Star Line chartered the cable ship Mackay - Bennett from Halifax , Nova Scotia to retrieve bodies . Three other ships followed in
the search : the cable ship Minia , the lighthouse supply ship Montmagny and the sealing vessel Algerine . Each ship left with embalming supplies , undertakers
, and clergy . Upon recovery , each body retrieved by the Mackay - Bennett was numbered and given as detailed a description as possible to help aid in
identification . The physical appearance of each body -- height , weight , age , hair and eye colour , visible birthmarks , scars or tattoos , was catalogued
and any personal effects on the bodies were gathered and placed in small canvas bags corresponding to their number .
</item 8>

<item 9>
Passengers of the RMS Titanic
1 British inquiry US inquiry Wreck of Titanic Maritime Memorial Act Deck officers Edward J. Smith ( Captain ) Henry Tingle Wilde ( Chief Officer ) William
McMaster Murdoch ( First Officer ) Charles H. Lightoller ( Second Officer ) Herbert Pitman ( Third Officer ) Joseph G. Boxhall ( Fourth Officer ) Harold G. Lowe
( Fifth Officer ) James Paul Moody ( Sixth Officer ) Joseph Bell ( Machine Room Manager ) Crew members Frederick Barrett Harold Bride William Denton Cox Sid
Daniels Alfred Frank Evans Frederick Fleet Luigi Gatti Robert Hichens Violet Jessop Charles Joughin Reginald Lee Evelyn Marsden William Mintram Jack Phillips
George Symons Passengers Fatalities Allison family Thomas Andrews John Jacob Astor IV David John Bowen Archibald Butt Thomas Byles Walter Donald Douglas Edith
Corse Evans Annie Funk Jacques Futrelle Sidney Leslie Goodwin Benjamin Guggenheim John Harper Wallace Hartley Charles Melville Hays Edward Austin Kent Joseph
Philippe Lemercier Laroche Francis Davis Millet Harry Markland Molson Michel Navratil Eino Viljami Panula W.T. Stead Ida Straus Isidor Straus John Borland
Thayer Jr . Frank M. Warren , Sr .
</item 9>

<item 10>
Passengers of the RMS Titanic
W.W. Norton & Company . p. 85 . ISBN 0 - 393 - 04782 - 2 . Jump up ^ Charles Pellegrino Web -- Edith Russell Jump up ^ Butler , Daniel Allen ( 1998 ) .
Unsinkable The Full Story of the RMS Titanic . Stackpole Books . pp. 200 -- 202 . ISBN 978 - 0 - 8117 - 1814 - 1 . Jump up ^ `` Maritime Museum of the Atlantic
Titanic Research Page -- Victims '' . Museum.gov.ns.ca. 8 November 2010 . Archived from the original on 7 December 2009 . Retrieved 29 January 2011 . Jump up ^
Mowbray , Jay Henry ( 1912 ) . `` CHAPTER XXI . THE FUNERAL SHIP AND ITS DEAD '' . The sinking of the Titanic ( 1912 ) . Archived from the original on 11
October 2007 . Retrieved 24 November 2008 . Jump up ^ Encyclopedia Titanica . `` Titanic Victims buried at Fairview Lawn Cemetery '' .
</item 10>
=== END ITEM id=443959ed30540c82 ===

=== ITEM id=e8b41c4917ee6de0 ===
Gold sentences and facts:
Sentence 0: R.H. Thomson as Matthew Cuthbert
  Fact 0.0: R.H. Thomson played the role of Matthew Cuthbert.

Retrieved items:
<item 1>
Anne (TV series)
of seasons No. of episodes 7 ( list of episodes ) Production Executive producer ( s ) Elizabeth Bradley Alex Sapot Sally Catto Debra Hayward Alison Owen Miranda
de Pencier Moira Walley - Beckett Producer ( s ) Susan Murdoch Cinematography Bobby Shore Running time 44 minutes Production company ( s ) Pelican Ballet
Northwood Entertainment Distributor Netflix Release Original network CBC Television Netflix ( worldwide ) Picture format 4K ( Ultra HD ) Audio format Dolby
Digital 5.1 with Descriptive Video Service track Original release March 19 , 2017 ( 2017 - 03 - 19 ) -- present ( present ) External links Website Anne ( titled
Anne with an E for international distribution on Netflix ) is a Canadian drama television series based on the 1908 novel Anne of Green Gables by Lucy Maud
Montgomery , and adapted by Emmy Award - winning writer and producer Moira Walley - Beckett . It airs on CBC in Canada , and elsewhere in the world it is
available for streaming on Netflix . The first season consists of seven episodes , with Niki Caro directing the 90 - minute season premiere .
</item 1>

<item 2>
Anne (TV series)
7 `` Wherever You Are Is My Home '' Amanda Tapping Moira Walley - Beckett April 30 , 2017 ( 2017 - 04 - 30 ) N / A The Cuthberts vow to do whatever it takes to
save the farm which reminds Anne of the strength of friendship and love . Reception ( edit ) Critical response ( edit ) The series has achieved a rating of 88 %
on Rotten Tomatoes , the site 's critical consensus stating : `` Anne with an E uses its complex central character to offer a boldly stylish , emotionally
resonant spin on classic source material that satisfies in its own right . '' The series has received a rating of 79 on Metacritic based on fifteen reviews ,
indicating `` generally favorable reviews '' . Emily Ashby , writing for Common Sense Media , calls the series an `` exceptional '' and `` spectacular ''
interpretation , giving it four out of five stars . Tasha Cerny , contributor for the Tracking Board , praises the cinematography as lush and colourful , the
characters vibrant , and the plot `` surprisingly thrilling for a story about a young girl living in a small secluded community in the late nineteenth century .
</item 2>

<item 3>
*Anne* (internationally titled *Anne with an E*) is a Canadian drama adapted by Moira Walley-Beckett from Lucy Maud Montgomery’s 1908 novel *Anne of Green
Gables*. It premiered on CBC on March 19, 2017, and streams internationally on Netflix. Its first season has seven episodes; Niki Caro directed the 90-minute
premiere. Episodes run about 44 minutes, with 4K video and Dolby Digital 5.1
</item 3>

<item 4>
*Anne* (*Anne with an E*), Moira Walley-Beckett’s CBC/Netflix adaptation of L.M. Montgomery’s *Anne of Green Gables*, premiered in March 2017. Amybeth McNulty
stars as orphan Anne, mistakenly sent to siblings Matthew and Marilla Cuthbert. Critics praised its cast, visuals, emotional depth, and attention to historical
hardship, but some objected to its darker tone, added tragedy, melodrama, and heavy-handed feminist changes, arguing
</item 4>

<item 5>
Anne (TV series)
When Matthew goes to pick the child up at the railway station , he finds not a boy , but a high - spirited and talkative girl , Anne Shirley . At first , the
Cuthberts are inclined to send her back , particularly after Marilla 's brooch goes missing , and Anne , in despair , runs away . Reconciled and settled in her
new home and school , Anne once again displays boundless enthusiasm which is nevertheless easily turned into despair when things go wrong , which they often do
. Slowly , her ebullient nature wins over those around her . Cast ( edit ) Main ( edit ) Amybeth McNulty as Anne Shirley Geraldine James as Marilla Cuthbert
R.H. Thomson as Matthew Cuthbert Dalila Bela as Diana Barry Lucas Jade Zumann as Gilbert Blythe Corrine Koslo as Rachel Lynde Aymeric Jett Montaz as Jerry
Baynard Others ( edit ) Jonathan Holmes as Mr. William Barry Helen Johns as Mrs. Eliza Barry Ryan Kiera Armstrong as Minnie Mary Barry Deborah Grover as
Josephine Barry Wayne Best as John Blythe Phillip Williams as Thomas Lynde David Ingram as Mr.
</item 5>

<item 6>
*Anne* premiered on CBC on March 19, 2017, and its first season ended April 30. It follows orphan Anne Shirley, mistakenly sent to siblings Matthew and Marilla
Cuthbert’s Green Gables farm instead of the boy they expected to help them. Netflix released the series as *Anne with an E* on May 12, 2017. CBC and Netflix
renewed it for a 10-episode second season, which began production in November 2017 and was
</item 6>

<item 7>
The reimagined *Anne of Green Gables* series, *Anne with an E*, was renewed by CBC and Netflix for a second season in August 2017, with episodes written by a
team of women. Coverage highlighted its darker, more feminist take on Anne Shirley. Casting announcements and behind-the-scenes reports followed filming in
Millbrook, Ontario.
</item 7>

<item 8>
*Anne* (also known as *Anne with an E*) is a Canadian English-language drama created and written by Moira Walley-Beckett, based on Lucy Maud Montgomery’s *Anne
of Green Gables*. It stars Amybeth McNulty, Geraldine James, and R.H. Thomson. In episode seven, the Cuthberts fight to save their farm, highlighting friendship
and love. Critics praised its emotional storytelling and visuals; it scored 88% on Rotten Tomatoes
</item 8>

<item 9>
Anne (TV series)
Anne ( TV series ) - wikipedia Anne ( TV series ) Jump to : navigation , search Anne Also known as Anne with an E Genre Drama Created by Moira Walley - Beckett
Based on Anne of Green Gables by Lucy Maud Montgomery Screenplay by Moira Walley - Beckett Starring Amybeth McNulty Geraldine James R.H. Thomson Lucas Jade
Zumann Dalila Bela Corrine Koslo Aymeric Jett Montaz Opening theme `` Ahead by a Century '' by The Tragically Hip Country of origin Canada Original language ( s
) English No. of seasons No.
</item 9>

<item 10>
Anne (TV series)
'' She went on to call Anne Shirley an `` accidental feminist '' , and how she `` really wanted to tell this story now . '' Broadcast and Release ( edit ) The
series initially premiered on March 19 , 2017 , on CBC and aired on a weekly basis , the season finale airing on April 30 , 2017 . The series debuted on Netflix
on May 12 , 2017 , under the title Anne with an E . Episodes ( edit ) Season 1 ( 2017 ) ( edit ) No . Title Directed by Written by Original air date Can .
viewers ( millions ) `` Your Will Shall Decide Your Destiny '' Niki Caro Moira Walley - Beckett March 19 , 2017 ( 2017 - 03 - 19 ) 0.999 When a miscommunication
brings a girl , Anne Shirley , to Green Gables instead of a boy , Marilla and Matthew Cuthbert are faced with a life - changing decision .
</item 10>
=== END ITEM id=e8b41c4917ee6de0 ===

=== ITEM id=62adec49ebbbc8a5 ===
Gold sentences and facts:
Sentence 0: Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher Usmanov ( 30.04 % ) 15 ! $22.4 B Kroenke Sports Enterprises Metal and mining , investment Bournemouth Maxim Demin ( 75 % ) PEAK6 Investments ( 25 % ) Petrochemicals trading Seven Developments Brighton & Hove Albion Tony Bloom ( 75.61 % ) 12 ! $1.7 B Online Gambling , Real Estate , Property , Land Development & Investments Burnley Mike Garlick ( 49.3 % ) John Banaszkiewicz ( 27.55 % ) 18 ! $80 M -- - Michael Bailey Associates Freight Investor Services Chelsea ( more information ) Roman Abramovich 03 ! $9.3 B Oil and industry Crystal Palace Steve Parish Martin Long Joshua Harris David S. Blitzer Jeremy Hosking Stephen Browett 15 ! $65 M $100 M $3.2 B $1.3 B $250 M $110 M Tag Worldwide Churchill Insurance Apollo Global Management Blackstone Group Marathon Asset Management Farr Vintners Everton ( more information ) Farhad Moshiri ( 49.9 % ) Bill Kenwright CBE Jon Woods 15 ! $2.3 B Steel and energy Theatre production Planet Hollywood Ocean Software Huddersfield Town Dean Hoyle 09 ! $475 M Card Factory Leicester City The Srivaddhanaprabha Family 10 ! $4.9 B King Power International Group Liverpool ( more information ) John W. Henry 11 ! $2.5 B Fenway Sports Group Manchester City ( more information ) Mansour bin Zayed Al Nahyan ( 87 % ) China Media Capital ( 13 % ) 01 ! $44 B Oil and industry investment Government Manchester United ( more information ) The Glazer Family 06 ! $4.9 B First Allied Corporation , Tampa Bay Buccaneers Newcastle United ( more information ) Mike Ashley 04 ! $3.3 B Sports Direct International Southampton Gao Jisheng ( 80 % ) Katharina Liebherr ( 20 % ) 07 ! $3.9 B + Lander Sports Development Co Ltd Inheritance Stoke City The Coates Family 08 ! $3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B -- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport , Sunday Sport Gold Group International Straumur Investment Bank
  Fact 0.0: Arsenal's owner E. Stanley Kroenke holds 67.05% of the club.
  Fact 0.1: Arsenal's owner Alisher Usmanov holds 30.04% of the club.
  Fact 0.2: Arsenal's estimated combined owner net worth is $22.4 B.
  Fact 0.3: Arsenal's owners' source of wealth is metal and mining and investment.
  Fact 0.4: Kroenke Sports Enterprises is listed for Arsenal.
  Fact 0.5: Bournemouth's owner Maxim Demin holds 75% of the club.
  Fact 0.6: Bournemouth's owner PEAK6 Investments holds 25% of the club.
  Fact 0.7: Bournemouth's source of wealth is petrochemicals trading.
  Fact 0.8: Seven Developments is listed for Bournemouth.
  Fact 0.9: Brighton & Hove Albion's owner Tony Bloom holds 75.61% of the club.
  Fact 0.10: Brighton & Hove Albion's owner net worth is $1.7 B.
  Fact 0.11: Brighton & Hove Albion's source of wealth is online gambling, real estate, property, land development and investments.
  Fact 0.12: Burnley's owner Mike Garlick holds 49.3% of the club.
  Fact 0.13: Burnley's owner John Banaszkiewicz holds 27.55% of the club.
  Fact 0.14: Burnley's estimated combined owner net worth is $80 M.
  Fact 0.15: Burnley's sources of wealth are Michael Bailey Associates and Freight Investor Services.
  Fact 0.16: Chelsea's owner is Roman Abramovich.
  Fact 0.17: Chelsea's owner net worth is $9.3 B.
  Fact 0.18: Chelsea's source of wealth is oil and industry.
  Fact 0.19: Crystal Palace's owners are Steve Parish, Martin Long, Joshua Harris, David S. Blitzer, Jeremy Hosking and Stephen Browett.
  Fact 0.20: Crystal Palace's owners' net worths are listed as $65 M, $100 M, $3.2 B, $1.3 B, $250 M and $110 M.
  Fact 0.21: Crystal Palace's sources of wealth are listed as Tag Worldwide, Churchill Insurance, Apollo Global Management, Blackstone Group, Marathon Asset Management and Farr Vintners.
  Fact 0.22: Everton's owner Farhad Moshiri holds 49.9% of the club.
  Fact 0.23: Everton's owners include Bill Kenwright CBE and Jon Woods.
  Fact 0.24: Everton's estimated combined owner net worth is $2.3 B.
  Fact 0.25: Everton's sources of wealth are steel and energy, theatre production, Planet Hollywood and Ocean Software.
  Fact 0.26: Huddersfield Town's owner is Dean Hoyle.
  Fact 0.27: Huddersfield Town's owner net worth is $475 M.
  Fact 0.28: Huddersfield Town's source of wealth is Card Factory.
  Fact 0.29: Leicester City's owner is The Srivaddhanaprabha Family.
  Fact 0.30: Leicester City's owner net worth is $4.9 B.
  Fact 0.31: Leicester City's source of wealth is King Power International Group.
  Fact 0.32: Liverpool's owner is John W. Henry.
  Fact 0.33: Liverpool's owner net worth is $2.5 B.
  Fact 0.34: Liverpool's source of wealth is Fenway Sports Group.
  Fact 0.35: Manchester City's owner Mansour bin Zayed Al Nahyan holds 87% of the club.
  Fact 0.36: Manchester City's owner China Media Capital holds 13% of the club.
  Fact 0.37: Manchester City's owner net worth is $44 B.
  Fact 0.38: Manchester City's sources of wealth are oil and industry investment and government.
  Fact 0.39: Manchester United's owner is The Glazer Family.
  Fact 0.40: Manchester United's owner net worth is $4.9 B.
  Fact 0.41: Manchester United's sources of wealth are First Allied Corporation and Tampa Bay Buccaneers.
  Fact 0.42: Newcastle United's owner is Mike Ashley.
  Fact 0.43: Newcastle United's owner net worth is $3.3 B.
  Fact 0.44: Newcastle United's source of wealth is Sports Direct International.
  Fact 0.45: Southampton's owner Gao Jisheng holds 80% of the club.
  Fact 0.46: Southampton's owner Katharina Liebherr holds 20% of the club.
  Fact 0.47: Southampton's owner net worth is $3.9 B.
  Fact 0.48: Southampton's source of wealth is Lander Sports Development Co Ltd and inheritance.
  Fact 0.49: Stoke City's owner is The Coates Family.
  Fact 0.50: Stoke City's owner net worth is $3.1 B.
  Fact 0.51: Stoke City's source of wealth is Bet365.
  Fact 0.52: Swansea City's owners Stephen Kaplan and Jason Levien hold 68% of the club.
  Fact 0.53: Swansea City's Supporters Trust holds 21.1% of the club.
  Fact 0.54: Swansea City's sources of wealth are Memphis Grizzlies, D.C. United and investment trust.
  Fact 0.55: Tottenham Hotspur's owner Joe Lewis holds 70.6% of the 85% holding company.
  Fact 0.56: Tottenham Hotspur's owner Daniel Levy holds 29.4% of the 85% holding company.
  Fact 0.57: Tottenham Hotspur's owner net worth is $5.6 B.
  Fact 0.58: Tottenham Hotspur's sources of wealth are Tavistock Group and ENIC Group.
  Fact 0.59: Watford's owner is Gino Pozzo.
  Fact 0.60: Watford's owner is described as a businessman.
  Fact 0.61: West Bromwich Albion's owner Guochuan Lai holds 87.8% of the club.
  Fact 0.62: West Bromwich Albion's owner net worth is $6.2 B.
  Fact 0.63: West Bromwich Albion's source of wealth is investment.
  Fact 0.64: West Ham United's owner David Sullivan holds 51.1% of the club.
  Fact 0.65: West Ham United's owner David Gold holds 35.1% of the club.
  Fact 0.66: West Ham United's owner Straumur Investment Bank holds 10% of the club.
  Fact 0.67: West Ham United's owners' net worths are listed as $520 M and $325 M.
  Fact 0.68: West Ham United's sources of wealth are Daily Sport, Sunday Sport, Gold Group International and Straumur Investment Bank.
Sentence 1: Club Owner ( s ) Estimated combined net worth ( £ m ) Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % ) Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % ) Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company Middlesbrough Steve Gibson 10 ! $260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn - Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C. Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 ! $155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment
  Fact 1.0: Aston Villa's owner is Dr. Tony Xia.
  Fact 1.1: Aston Villa's owner net worth is $1.2 B.
  Fact 1.2: Aston Villa's source of wealth is Recon Group.
  Fact 1.3: Barnsley's owner is Patrick Cryne.
  Fact 1.4: Barnsley's owner net worth is $40 M.
  Fact 1.5: Barnsley's source of wealth is iSOFT.
  Fact 1.6: Birmingham City's owners are Paul Suen (60.78%), Wang Lei (5.28%) and Carson Yeung (6.51%).
  Fact 1.7: Birmingham City's estimated combined owner net worth is $570 M.
  Fact 1.8: Birmingham City's source of wealth is Birmingham International Holdings.
  Fact 1.9: Bolton Wanderers' owner is Eddie Davies CBE.
  Fact 1.10: Bolton Wanderers' owner net worth is $52 M.
  Fact 1.11: Bolton Wanderers' source of wealth is Strix.
  Fact 1.12: Brentford's owner Matthew Benham holds 96% of the club.
  Fact 1.13: Bristol City's owner is Stephen Lansdown.
  Fact 1.14: Bristol City's owner net worth is $1.9 B.
  Fact 1.15: Bristol City's source of wealth is Hargreaves Lansdown.
  Fact 1.16: Burton Albion's owner is Ben Robinson.
  Fact 1.17: Cardiff City's owner Vincent Tan holds 51% of the club.
  Fact 1.18: Cardiff City's owner net worth is $820 M.
  Fact 1.19: Cardiff City's source of wealth is Berjaya Group.
  Fact 1.20: Derby County's owner is Mel Morris.
  Fact 1.21: Derby County's owner net worth is $520 M.
  Fact 1.22: Fulham's owner is Shahid Khan.
  Fact 1.23: Fulham's owner net worth is $8.9 B.
  Fact 1.24: Fulham's source of wealth is Flex-N-Gate Corp, an automobile parts manufacturer.
  Fact 1.25: Hull City's owner is The Allam Family.
  Fact 1.26: Hull City's owner net worth is $325 M.
  Fact 1.27: Hull City's source of wealth is Allam Marine.
  Fact 1.28: Ipswich Town's owner is Marcus Evans.
  Fact 1.29: Ipswich Town's owner net worth is $825 M.
  Fact 1.30: Ipswich Town's source of wealth is Marcus Evans Group.
  Fact 1.31: Leeds United's owners are Andrea Radrizzani, Eleonora Sports Ltd and Aser Group Company.
  Fact 1.32: Middlesbrough's owner is Steve Gibson.
  Fact 1.33: Middlesbrough's owner net worth is $260 M.
  Fact 1.34: Middlesbrough's source of wealth is Bulkhaul Limited.
  Fact 1.35: Millwall's owners are John Berylson and Richard Smith.
  Fact 1.36: Millwall's owner net worth is $175 M.
  Fact 1.37: Millwall's source of wealth is Chestnut Hill Ventures.
  Fact 1.38: Norwich City's owners Delia Smith and Michael Wynn-Jones hold 53% of the club.
  Fact 1.39: Norwich City's owner Michael Foulger holds 15% of the club.
  Fact 1.40: Norwich City's owner net worth is $30 M.
  Fact 1.41: Norwich City's sources of wealth are food industry, publishing and poultry.
  Fact 1.42: Nottingham Forest's owner is Evangelos Marinakis.
  Fact 1.43: Nottingham Forest's owner net worth is $650 M.
  Fact 1.44: Nottingham Forest's owner is also owner of Olympiacos F.C.
  Fact 1.45: Preston North End's owner Trevor Hemmings CVO holds 51.4% of the club.
  Fact 1.46: Preston North End's owner net worth is $720 M.
  Fact 1.47: Preston North End's sources of wealth are Blackpool Tower, horse racing, investment, Northern Trust and Sportech.
  Fact 1.48: Queens Park Rangers' owner Tony Fernandes holds 66% of the club.
  Fact 1.49: Queens Park Rangers' owner Lakshmi Mittal holds 33% of the club.
  Fact 1.50: Queens Park Rangers' owners' net worths are listed as $345 M and $16.4 B.
  Fact 1.51: Queens Park Rangers' sources of wealth are Air Asia, Tune Group and ArcelorMittal.
  Fact 1.52: Reading's owners are Narin Niruttinanon (50%), Sasima Srivikorn (25%) and Sumrith Thanakarnjanasuth (25%).
  Fact 1.53: Sheffield United's owner Abdullah bin Musa'ed holds 50% of the club.
  Fact 1.54: Sheffield United's owner net worth is $155 M.
  Fact 1.55: Sheffield United's sources of wealth are property and oil.
  Fact 1.56: Sheffield Wednesday's owner is Dejphon Chansiri.
  Fact 1.57: Sheffield Wednesday's source of wealth is Thai Union Group.
  Fact 1.58: Sunderland's owner is Ellis Short.
  Fact 1.59: Sunderland's owner net worth is $2.9 B.
  Fact 1.60: Sunderland's sources of wealth are Lone Star Funds and Skibo Castle.
  Fact 1.61: Wolverhampton Wanderers' owners are Fosun International (parent company) and Guo Guangchang (co-owner of Fosun).
  Fact 1.62: Wolverhampton Wanderers' owner net worth is $6.2 B.
  Fact 1.63: Wolverhampton Wanderers' source of wealth is investment.

Retrieved items:
<item 1>
List of owners of English football clubs
Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % )
Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % )
Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby
County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam
Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company
Middlesbrough Steve Gibson 10 !
</item 1>

<item 2>
List of owners of English football clubs
$250 M DW Sports Fitness , JJB Sports League Two ( edit ) Club Owner ( s ) Estimated combined net worth Source of wealth Accrington Stanley The Styring Family (
12 % ) Accrington Stanley Supporters Fund ( 12 % ) Peter Marsden ( 10 % ) Investment trust Barnet Blackpool Owen Oyston ( 80 % ) Valeri Belokon ( 25 % ) 07 !
$105 M $195 M Publishing , Media & Estate Agency Finance , Media , Food Industry & Mining Cambridge United Grosvenor Fund Management Grosvenor Group Owned by
the Duke of Westminster and family Property Carlisle United ( more information ) A. Jenkins ( 37 % of 93 % holding company ) J. Nixon ( 18 % ) S. Pattison ( 18
% ) The United Trust ( 25.4 % ) Pioneer Foods Businessman ( retired ) Carlisle Glass Investment Trust Cheltenham Town Paul Baker Chesterfield Dave Allen
Colchester United Robbie Cowling 09 !
</item 2>

<item 3>
List of owners of English football clubs
$22.4 B Kroenke Sports Enterprises Metal and mining , investment Bournemouth Maxim Demin ( 75 % ) PEAK6 Investments ( 25 % ) Petrochemicals trading Seven
Developments Brighton & Hove Albion Tony Bloom ( 75.61 % ) 12 ! $1.7 B Online Gambling , Real Estate , Property , Land Development & Investments Burnley Mike
Garlick ( 49.3 % ) John Banaszkiewicz ( 27.55 % ) 18 ! $80 M -- - Michael Bailey Associates Freight Investor Services Chelsea ( more information ) Roman
Abramovich 03 ! $9.3 B Oil and industry Crystal Palace Steve Parish Martin Long Joshua Harris David S. Blitzer Jeremy Hosking Stephen Browett 15 ! $65 M $100 M
$3.2 B $1.3 B $250 M $110 M Tag Worldwide Churchill Insurance Apollo Global Management Blackstone Group Marathon Asset Management Farr Vintners Everton ( more
information ) Farhad Moshiri ( 49.9 % ) Bill Kenwright CBE Jon Woods 15 !
</item 3>

<item 4>
List of owners of English football clubs
Jump up ^ `` Shahid Khan '' . Forbes . Forbes.com LLC. 19 May 2017 . Retrieved 19 May 2017 . Jump up ^ `` Who 's Who At The KC '' . Hull City F.C. 25 June 2012
. Retrieved 9 May 2014 . Jump up ^ `` Sunday Times Rich List 2017 reveals Yorkshire 's 20 wealthiest people '' . Hull Daily Mail . Archived from the original on
2 May 2017 . Retrieved 19 May 2017 . Jump up ^ www.calcioefinanza.it Leeds http://www.calcioefinanza.it/2017/01/04/leeds-cellino-radrizzani/ Leeds Check url =
value ( help ) . Retrieved 2017 - 01 - 10 . Missing or empty title = ( help ) Jump up ^ leeds united
https://www.leedsunited.com/news/club-news/21921/club-statement . Retrieved 2017 - 05 - 23 .
</item 4>

<item 5>
List of owners of English football clubs
List of owners of English football clubs - wikipedia List of owners of English football clubs Jump to : navigation , search This is a list of the current owners
of English football clubs , as well as ( in some cases ) their estimated net worth and source of wealth . All figures are listed in millions rather than
billions . For example , £ 1,000 m would be one billion ( 1,000,000,000 ) pounds . Only shareholders with a significant interest ( above 10 % ) are listed .
Contents ( hide ) 1 Premier League 2 Football League Championship 3 League One 4 League Two 5 References Premier League ( edit ) This list is incomplete ; you
can help by expanding it . Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher
Usmanov ( 30.04 % ) 15 !
</item 5>

<item 6>
List of owners of English football clubs
$155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton
Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment League One ( edit ) Club Owner ( s ) Estimated combined
net worth Source of wealth AFC Wimbledon Wimbledon Football Club Supporters ' Society ( The Dons Trust ) Supporters Trust Blackburn Rovers Balaji & Venkatesh
Rao ( 99.9 % ) 04 ! $650 M Venkateshwara Hatcheries Group Bradford City Bristol Rovers Dwane Sports ( 92 % ) Banking and tourism Bury Save Our Shakers Trust (
63.8 % ) The Bury F.C. Supporters Society Ltd ( Forever Bury ) ( 11 % ) Investment trusts Charlton Athletic Roland Duchatelet ( 95 % ) 24 ! Staprix NV Doncaster
Rovers John Ryan Dick Watson ( 26 % ) Terry Bramall CBE 07 !
</item 6>

<item 7>
List of owners of English football clubs
$3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United
Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B
-- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment
West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport ,
Sunday Sport Gold Group International Straumur Investment Bank Football League Championship ( edit ) Club Owner ( s ) Estimated combined net worth ( £ m )
Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B !
</item 7>

<item 8>
List of owners of English football clubs
$260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn -
Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C.
Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more
information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin
Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 !
</item 8>

<item 9>
List of owners of English football clubs
Brentford F.C . Retrieved 29 May 2014 . Jump up ^ `` # 1098 Stephen Lansdown '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` Club Ownership '' . Cardiff City
F.C . Retrieved 30 November 2013 . Jump up ^ `` Forbes ' Billionaires '' . Forbes . Jump up ^ `` Club Statement : Mel Morris Completes Purchase Of Derby County
'' . Derby County F.C. 4 September 2015 . Retrieved 4 September 2015 . Jump up ^ `` Sunday Times Rich List : Candy Crush tycoon Mel Morris and Duke of
Devonshire among Derbyshire 's millionaires '' . Derby Telegraph . 26 April 2015 . Archived from the original on 3 May 2015 . Retrieved 26 April 2015 . Jump up
^ `` Welcome To Shahid Khan '' . Fulham FC. 12 July 2013 . Retrieved 12 July 2013 . Jump up ^ `` Shahid Khan '' . Forbes .
</item 9>

<item 10>
List of owners of English football clubs
`` Trevor Hemmings ' Northern Trust returns to profit '' . The Daily Telegraph . London . Retrieved 12 February 2013 . Jump up ^ `` Company details '' . Queens
Park Rangers F.C. 10 July 2012 . Archived from the original on 10 August 2012 . Retrieved 12 February 2012 . Jump up ^ `` # 28 Anthony Fernandes '' . Forbes .
Retrieved 27 May 2014 . Jump up ^ `` # 56 Lakshmi Mittal '' . Forbes . Retrieved 19 May 2017 . Jump up ^ Details , Company . `` COMPANY DETAILS '' . Retrieved
29 July 2016 . Jump up ^ `` Fantasy football nut Saudi Prince Abdullah is joint owner of League One side Sheffield United '' . Daily Mail . London . 3 September
2013 . Retrieved 5 September 2013 . Jump up ^ `` Board & Management '' . A.F.C. Sunderland . Retrieved 9 May 2014 .
</item 10>
=== END ITEM id=62adec49ebbbc8a5 ===

=== ITEM id=602cd316b268d682 ===
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
The base model receives body - color exterior mirrors ( previously black ) , standard security system and new wheel covers for 2012 . Canadian trims include the
DX ( two speakers , black door handles and side mirrors ) , DX - A ( adds factory - installed air conditioning ) , LX ( adds among other things , body - colored
door handles and side mirrors , 15 - inch lightweight alloy wheels from the 2006 -- 11 Civic Hybrid and DX - G , rear spoiler , power locks and remote entry ,
two speakers for a total of four , cruise control ) , and Fit Sport ( adds USB jack and Vehicle Stability Assist ) . The Fit Sport is similar to its American
counterpart , however a navigation system is not offered as an option . There are plans to build subcompact cars , which many rumoured to be Fit , in Honda 's
new plant in Mexico . NHTSA Crash Test Ratings : Frontal Driver and Passenger : Side Driver Impact : Side Rear Passenger Impact : Rollover : Brazil The Sumare -
built new Fit was introduced in October 2008 in Brazil at the Sao Paulo Auto Show .
</item 1>

<item 2>
Honda Fit
With the fuel tank placed in the center plus an ultra-thin electric motor and compact IPU , it claims to achieve an efficient weight distribution . In an
interview , Kohei Hitomi , the project leader of Fit hybrid , admitted that lowering CO further will affect some attributes that buyers of the car value : a
bigger electric motor and battery sacrifice spaces utility , and improving aerodynamics might make side mirrors smaller . He said there are benefits for
customers to keep all the versatility . The drivetrain is the same drive system as the Honda Insight . The advantages of the system are low cost , low weight
and that it can be combined with either manual transmission or continuous variable transmission . The car is better suited than its diesel competitors in areas
where superminis are most common : city centres . It is more economical in urban conditions . Thanks to the instant torque of the electric motor , it has a much
better engine response . The hybrid version was not available in the U.S. According to Koichi Kondo , Executive Vice President of Honda : `` As for the future ,
it 's open to question . We will carefully be watching the market situation . ''
</item 2>

<item 3>
Honda Fit
In addition , the Fit Sport now comes with carpeted floor mats . As of the 2011 model year , the base Fit comes with a remote entry , 15 - inch wheels with
covers , power windows , power locks , cruise control , a tilt and telescoping steering wheel , a 160 - watt AM / FM / CD audio system with four speakers , a
USB audio interface , a 12 - volt power outlet , and other features . The Fit Sport adds a ground appearance package , 16 - inch ( 410 mm ) alloy wheels ,
paddle shifters , a navigation system , a security system , fog lights , chrome exhaust finisher , a leather - wrapped steering wheel , steering wheel - mounted
audio controls , floor mats , a 6 - speaker AM / FM / CD audio system , and other features . 2010 Honda Fit Sport 2012 Honda Fit Sport ( US ) For 2012 , the Fit
features revised floor , front fenders , A-pillars , and front corner - window glass . The Fit Sport features a revised grille , dark `` machined surface ''
finish alloy wheels , a black monochrome theme interior with chrome accents .
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
Indonesia Santa Rosa , Philippines Guangzhou , China ( Guangqi Honda ) Sumaré , Brazil ( Honda Brazil ) Designer Mitsuhiro Honda ( 1999 ) Body and chassis
Related Honda Airwave Honda City / Fit Aria Powertrain Engine 1.2 L L12A I4 ( petrol ) 1.3 L L13A I4 ( petrol ) 1.5 L L15A I4 ( petrol ) Transmission 5 - speed
manual 5 - speed automatic CVT automatic Dimensions Wheelbase 2,450 mm ( 96.5 in ) Length 3,845 mm ( 151.4 in ) ( Worldwide ) 4,000 mm ( 157.4 in ) ( U.S. &
Canada ) Width 1,675 mm ( 65.9 in ) Height 1,525 mm ( 60.0 in ) Curb weight 1,084 kg ( 2,390 lb ) The Honda Fit debuted in June 2001 in Japan and immediately
became a big hit . At its introduction in 2001 , it won the Car of the Year Japan Award .
</item 5>

<item 6>
Honda Fit
Malaysia Guangzhou , China ( Guangqi Honda ) Celaya , Guanajuato , Mexico ( from Feb 2014 ) Tapukara , India ( from Jan 2015 ) Designer Riku Wada ( 2010 ) Body
and chassis Related Honda City Honda HR - V Honda Freed Honda Shuttle Acura CDX Powertrain Engine 1.2 L L12B i - VTEC I4 petrol 1.3 L L13B Earth Dreams i - VTEC
I4 1.5 L L15A I4 ( petrol ) 1.5 L L15B1 I4 ( petrol ) 1.5 L L15B Earth Dreams i - VTEC I4 1.5 L i - DTEC diesel Transmission 5 - speed manual ( China Only ) 6 -
speed manual 7 - speed semi-automatic ( DCT ) CVT automatic Dimensions Wheelbase 2,530 mm ( 99.6 in ) Length 3,955 mm ( 155.7 in ) Width 1,695 mm ( 66.7 in )
Height 1,525 -- 1,550 mm ( 60.0 -- 61.
</item 6>

<item 7>
Honda Fit
All models feature ECON Mode for enhancing real - world fuel economy . Hybrid model has Eco Assist ( Ecological Drive Assist System ) to help the driver engage
in fuel - efficient driving practices . It went on sale on June 16 , 2011 , in Japan . The Fit Shuttle comes with a double - hinged two - piece load floor that
both allows access to an under - floor storage bin and also functions as a divider for the cargo area . Through better sound dampening and insulation , the Fit
Shuttle is pledged to be as quiet as a mid-size sedan . Japan JDM - spec interior Japanese sales began in October 2007 . Facelifted models , and the hybrid
version , arrived in October 2010 . Models available in Japan are the 1.3 L and 1.5 L variants consisting of the 13G , 15XH and the top - of - the - line RS .
In 2007 , aftermarket tuner Mugen , introduced a new line of accessories for the Japanese market 2008 Honda Fit . On 17 January 2008 , they introduced the Mugen
Fit F154SC at the 2008 Tokyo Auto Salon .
</item 7>

<item 8>
Honda Fit
Gawker Media . Retrieved 2008 - 03 - 19 . Jump up ^ Bell , Damon . `` The Consumer Guide to 2009 Subcompact Cars '' . Consumer Guide Automotive . HowStuffWorks
. Archived from the original on 2009 - 08 - 30 . Retrieved 2009 - 08 - 25 . The most expensive subcompact , the top - trim - level Honda Fit with an optional
navigation system , has a still - modest base price of $18,760 . Jump up ^ `` Honda Fit production could shift to Ohio '' . The Columbus Dispatch . Archived
from the original on 2013 - 01 - 21 . Retrieved 2009 - 10 - 22 . Jump up ^ `` 2011 Honda Fit Receives More Standard Features ; VSA Added to All Models '' .
theautochannel.com . Jump up ^ `` 2009 Honda Fit Review '' . JB car pages . Retrieved 2009 - 03 - 09 . Jump up ^ `` 2012 Honda Fit Receives Updates and More
Features '' ( Press release ) .
</item 8>

<item 9>
Honda Fit
3 L L13A I4 ( petrol ) 1.5 L L15A I4 ( petrol ) Transmission 5 - speed manual 6 - speed manual 5 - speed automatic CVT automatic ( Japan ) Dimensions Wheelbase
2,500 mm ( 98.4 in ) Length 3,985 mm ( 156.9 in ) ( 2009 -- 11 ) 4,100 mm ( 161.4 in ) ( 2012 -- 13 ) 4,410 mm ( 173.6 in ) ( Fit Shuttle ) Width 1,695 mm (
66.7 in ) Height 1,525 mm ( 60.0 in ) 1,540 mm ( 60.6 in ) ( Fit Shuttle ) Curb weight 1,070 kg ( 2,359 lb ) The second generation Fit debuted on 17 October
2007 at the Tokyo Motor Show . At its introduction in 2007 , it won the Car of the Year Japan Award for the second time . The vehicle offered a longer wheelbase
than its predecessor and is wider and longer overall . Overall height is unchanged , while interior height increased by . 3 in .
</item 9>

<item 10>
Honda Fit
National Highway Traffic Safety Administration ( NHTSA ) crash test ratings . The engine is a 1.5 L ( 92 cu in ) i - VTEC 4 - cylinder with 117 horsepower at
6600 rpm mated to a five - speed manual transmission or a five - speed automatic transmission . The second generation Honda Fit went on sale in North America
around August 26 , 2008 , for the 2009 model year . It was considered the most expensive within the subcompact class when equipped as such , but was also
described as modestly priced . It was reported in September 2009 that Honda is considering producing the Fit in North America in response to the increasing
strength of the Japanese yen against the U.S. dollar . For the 2011 models , Vehicle Stability Assist , cruise control , remote entry and USB audio interface
became standard across all trim levels . In addition , the Fit Sport now comes with carpeted floor mats .
</item 10>
=== END ITEM id=602cd316b268d682 ===
