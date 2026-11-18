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

=== ITEM id=df451c7dd5ce36c3 ===
Gold sentences and facts:
Sentence 0: Bay of Montevideo
  Fact 0.0: Bay of Montevideo

Retrieved items:
<item 1>
Dead Sea
Peninsula Orinoco Delta South Tierra del Fuego Patagonia Pampas Pantanal Gran Chaco Chiquitano dry forests Valdes Peninsula West Andes Tropical Andes Wet Andes
Dry Andes Pariacaca mountain range Altiplano Atacama Desert Latin Hispanic American Cordillera Ring of Fire LAC Polar regions Antarctic Antarctic Peninsula East
Antarctica West Antarctica Eklund Islands Ecozone Extreme points Islands Arctic Arctic Alaska British Arctic Territories Canadian Arctic Archipelago Finnmark
Greenland Northern Canada Northwest Territories Nunavik Nunavut Russian Arctic Sakha Sápmi Yukon North American Arctic Earth 's oceans and seas Arctic Ocean
Amundsen Gulf Barents Sea Beaufort Sea Chukchi Sea East Siberian Sea Greenland Sea Gulf of Boothia Kara Sea Laptev Sea Lincoln Sea Prince Gustav Adolf Sea
Pechora Sea Queen Victoria Sea Wandel Sea White Sea Atlantic Ocean Adriatic Sea Aegean Sea Alboran Sea Archipelago Sea Argentine Sea Baffin Bay Balearic Sea
Baltic Sea Bay of Biscay Bay of Bothnia Bay of Campeche Bay of Fundy Black Sea Bothnian Sea Caribbean Sea Celtic Sea English Channel Foxe Basin Greenland Sea
Gulf of Bothnia Gulf of
</item 1>

<item 2>
Uvea
Uvea - wikipedia Uvea Jump to : navigation , search For the Pacific island and chiefdom in Wallis and Futuna , see Wallis ( island ) and Uvea ( Wallis and
Futuna ) . For the Pacific commune and island in New Caledonia , see Ouvéa and Ouvéa Island . This article needs additional citations for verification . Please
help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( February 2014 ) ( Learn how and when to
remove this template message ) Uvea Horizontal section of the eyeball . The constituents of the uvea follow : iris labeled at top , ciliary body labeled at
upper right , and choroid labeled at center right . ) Details Identifiers Latin tunica vasculosa bulbi MeSH D014602 TA A15. 2.03. 001 FMA 58103 Anatomical
terminology ( edit on Wikidata ) The uvea ( / ˈjuːvɪə / ) ( Lat .
</item 2>

<item 3>
Category:Bodies of water of Uruguay
Category : Bodies of water of Uruguay - wikipedia Help Category : Bodies of water of Uruguay Jump to : navigation , search Wikimedia Commons has media related
to Bodies of water in Uruguay . This category has the following 4 subcategories , out of 4 total . ► Bays of Uruguay ‎ ( 1 P ) ► Lakes of Uruguay ‎ ( 5 P ) ►
Rivers of Uruguay ‎ ( 2 C , 35 P ) ► Springs of Uruguay ‎ ( 1 P ) Pages in category `` Bodies of water of Uruguay '' This category contains only the following
page . This list may not reflect recent changes ( learn more ) . Bay of Montevideo Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Category:Bodies_of_water_of_Uruguay&oldid=756422847 '' Categories : Landforms of Uruguay Bodies of water by country
Water in Uruguay Bodies of water of South America by country Talk Help About Wikipedia Azərbaycanca Deutsch فارسی 한국어 Magyar Português Türkçe اردو Edit links
This page was last edited on 24 December 2016 , at 03 : 52 . About Wikipedia
</item 3>

<item 4>
FIFA World Cup hosts
Retrieved 15 March 2018 . Jump up ^ ElPais . `` Valdez : `` La candidatura original es Uruguay - Argentina '' `` . Retrieved 15 March 2018 . Jump up ^ Barreiro
, Ramiro ( 2 September 2017 ) . `` Messi y Suárez apadrinan el Mundial Uruguay Argentina 2030 '' . EL PAÍS . Retrieved 15 March 2018 . Jump up ^
https://www.thesun.co.uk/sport/football/4901830/england-amazingly-backed-by-germany-in-world-cup-2030-bid/ Jump up ^ `` Singapore Strong ' Host ' Candidate for
Joint 2034 World Cup Finals Bid , '' Today , ( 14 November 2017 ) ,
http://www.todayonline.com/sports/football/singapore-strong-host-candidate-joint-2034-world-cup-finals-bid Jump up ^ `` ASEAN mulling 2034 World Cup bid :
Myanmar '' . Retrieved 15 March 2018 . Jump up ^ hermes ( 6 July 2017 ) .
</item 4>

<item 5>
FIFA World Cup hosts
Retrieved 14 January 2017 . Jump up ^ `` UEFA will ' fight ' for Europe 's right to host 2030 World Cup '' . Retrieved 2 June 2017 . ^ Jump up to : Molestia en
Uruguay por video de Conmebol ( in Spanish ) Jump up ^ ElPais . `` Chile no se rinde con ser sede '' ( in Spanish ) . Retrieved 15 March 2018 . Jump up ^ Dudko
, James ( 29 July 2017 ) . `` Argentina , Uruguay Announce Bid to Host 2030 FIFA World Cup '' . Bleacher Report . Turner Broadcasting System . Jump up ^ Gadd ,
Mick ( 1 September 2017 ) . `` Lionel Messi and Luis Suarez campaign for Uruguay and Argentina 's 2030 World Cup bid before 0 - 0 draw '' . Daily Mirror . Jump
up ^ `` Es oficial : Paraguay se suma a la candidatura de Argentina y Uruguay para el Mundial 2030 '' ( It 's official : Paraguay adds itself to the Argentina
and Uruguay bid ) ( in Spanish ) .
</item 5>

<item 6>
Dead Sea
New Caledonia Solomon Islands ( archipelago ) Vanuatu Kula Gulf Australia Capital Country Eastern Australia Lake Eyre basin Murray -- Darling basin Northern
Australia Nullarbor Plain Outback Southern Australia Maralinga Sunraysia Great Victoria Desert Gulf of Carpentaria Gulf St Vincent Lefevre Peninsula Fleurieu
Peninsula Yorke Peninsula Eyre Peninsula Mornington Peninsula Bellarine Peninsula Mount Henry Peninsula Melanesia Islands Region Bismarck Archipelago Solomon
Islands Archipelago Fiji New Caledonia Papua New Guinea Vanuatu Micronesia Caroline Islands Federated States of Micronesia Palau Guam Kiribati Marshall Islands
Nauru Northern Mariana Islands Wake Island Polynesia Easter Island Hawaiian Islands Cook Islands French Polynesia Austral Islands Gambier Islands Marquesas
Islands Society Islands Tuamotu Kermadec Islands Mangareva Islands Samoa Tokelau Tonga Tuvalu Ring of Fire Regions of South America East Amazon basin Atlantic
Forest Caatinga Cerrado North Caribbean South America West Indies Los Llanos The Guianas Amazon basin Amazon rainforest Gulf of Paria Paria Peninsula Paraguaná
Peninsula Orinoco Delta South Tierra del Fuego Patagonia Pampas Pantanal Gran
</item 6>

<item 7>
FIFA World Cup hosts
The original bid logo for the Uruguay - Argentina World Cup Bid . Confirmed 2030 bids : CONMEBOL : Uruguay / Argentina / Paraguay : Uruguay -- Argentina --
Paraguay 2030 FIFA World Cup bid Expressed interest in bidding : UEFA : England : England 2030 FIFA World Cup bid 2034 FIFA World Cup ( edit ) Main article :
2034 FIFA World Cup The first bid for the 2034 FIFA World Cup has been proposed as a collective bid by the members of the Association of Southeast Asian Nations
. The idea of a combined ASEAN bid had been mooted as early as January 2011 , when the Football Association of Singapore 's then - President , Zainudin Nordin ,
said in a statement that the proposal had been made at an ASEAN Foreign Ministers meeting . In 2013 , Nordin and Special Olympics Malaysia President , Datuk
Mohamed Feisol Hassan , recalled the idea for ASEAN to jointly host a World Cup . Under FIFA rules as of 2017 , the 2030 World Cup can not be held in Asia ( AFC
) as Asian Football Confederation members are excluded from the bidding following the selection of Qatar in 2022 .
</item 7>

<item 8>
FIFA World Cup hosts
Under FIFA rules as of 2017 , the 2030 World Cup can not be held in Asia ( AFC ) because the Asian Football Confederation is excluded from the bidding following
the selection of Qatar in 2022 . Also in June 2017 , UEFA 's president Aleksander Čeferin stated that Europe ( UEFA ) will definitely fight for its right to
host the 2030 World Cup . The Uruguay -- Argentina bid would coincide with the centennial anniversary of the first FIFA World Cup final , and the bicentennial
of the first Constitution of Uruguay . The Uruguay - Argentina bid was officially confirmed on 29 July 2017 . Paraguay is aiming for joining the bid , but this
is not confirmed or discarded yet . The efforts for the inclusion of Paraguay in the bid caused discomfort in the Uruguayan Football Association , and Wilmar
Valdéz ( AUF president ) accused some insistence of the Conmebol although it is not official that Paraguay joins the event . Chile also wants to organize the
World Cup with Uruguay and Argentina . A joint bid was announced by the Argentine Football Association and the Uruguayan Football Association on 29 July 2017 .
</item 8>

<item 9>
Pan-American Highway
Other branches probably exist across the center of South America . Almost all Pan-American sections in Gran Buenos Aires are modern and fast main highways West
Access to Buenos Aires Brazil and Uruguay ( edit ) A continuation of the Pan-American Highway to the Brazilian cities of São Paulo and Rio de Janeiro uses a
ferry from Buenos Aires to Colonia in Uruguay and Uruguay Highway 1 to Montevideo . Uruguay Highway 9 and Brazil Highway 471 route to near Pelotas , from where
Brazil Highway 116 leads to Brazilian main cities . Guyana , Suriname and French Guyana ( edit ) The highway does not have official segments to Belize , Guyana
, Suriname and French Guiana , nor to any of the island nations in the Americas . However , highways from Venezuela link to Brazilian Trans - Amazonian highway
that provides a southwest entrance to Guyana , route to the coast , and follow a coastal route through Suriname to French Guiana . West Indies section ( edit )
Plans have been discussed for including the West Indies in the Pan American Highway system . According to these , a system of ferries would be established to
connect terminal points of the highway .
</item 9>

<item 10>
FIFA World Cup hosts
Before Uruguay and Argentina played out a goalless draw in Montevideo , FC Barcelona players Luis Suárez and Lionel Messi promoted the bid with commemorative
shirts . On 31 August 2017 , it was suggested Paraguay would join as a third host . CONMEBOL , the South American confederation , confirmed the joint three -
way bid in September 2017 . English FA vice chairman David Gill has proposed that his country could potentially bid for 2030 , provided that the bidding process
is made more transparent . `` England is one of few countries that could stage even a 48 - nation event in its entirety , while Football Association chief
executive Martin Glenn made it clear earlier this year bidding for 2030 was an option . '' In June 2017 , UEFA stated that `` it would support a pan-British bid
for 2030 or even a single bid from England . '' Moreover , a possible English bid for 2030 was also backed by the German Football Association . It is expected
that countries from `` South America ( CONMEBOL ) and Europe ( UEFA ) will likely to be in a bid competition for 2030 . ''
</item 10>
=== END ITEM id=df451c7dd5ce36c3 ===

=== ITEM id=57ea7058b182e4ab ===
Gold sentences and facts:
Sentence 0: `` The Joker '' Cover of the 1973 single Single by Steve Miller Band from the album The Joker B - side `` ' see track listings ' '' Released October 1973 Format 7 '' Single , 12 '' single , Cassette single ( Original Releases ) , CD single and Digital download ( Post Releases ) . Genre Rock , blues rock , southern rock Length 4 : 26 ( Album version ) ; 3 : 35 ( 7 '' version / Radio edit ) and 6 : 20 ( Extended version ) Label Capitol Records ; EMI - Odeon ; London Records , Parlophone Records , EMI Bovema , Odeon Records . Songwriter ( s ) Eddie Curtis , Ahmet Ertegün , Steve Miller Producer ( s ) Steve Miller Steve Miller Band singles chronology `` Fandango '' ( 1972 ) `` The Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash '' ( 1974 ) `` Fandango '' ( 1972 ) `` The Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash '' ( 1974 ) `` Take the Money and Run '' / `` The Joker '' ( 1983 , double A-side , live ) Cover of the 1983 live single Steve Miller Band singles chronology `` Living in the U.S.A. '' ( live ) ( 1983 ) String Module Error : Match not found1983 `` Take the Money and Run '' ( live ) / `` The Joker '' ( live ) ( 1983 ) String Module Error : Match not found1983 `` Shangri - La '' ( 1984 ) Shangri - La1984 `` The Joker '' ( 1990 , reissue ) Cover of the 1990 reissue Steve Miller Band singles chronology `` Ya Ya '' ( 1988 ) Ya Ya1988 `` The Joker '' ( 1990 ) The Joker1990 `` Wide River '' ( 1993 ) Wide River1993
  Fact 0.0: The Joker's 1973 single was released in October 1973 by Steve Miller Band.
  Fact 0.1: The 1973 single 'The Joker' is from the album The Joker.
  Fact 0.2: The Joker single was released in 7-inch single, 12-inch single, and cassette single formats (original releases).
  Fact 0.3: The Joker single was released as a CD single and digital download (later releases).
  Fact 0.4: The Joker single's genres are rock, blues rock, and southern rock.
  Fact 0.5: The Joker single's album version is 4:26 long.
  Fact 0.6: The Joker single's 7-inch version / radio edit is 3:35 long.
  Fact 0.7: The Joker single's extended version is 6:20 long.
  Fact 0.8: The Joker single's label is Capitol Records.
  Fact 0.9: The Joker single was also issued on EMI-Odeon, London Records, Parlophone Records, EMI Bovema, and Odeon Records labels.
  Fact 0.10: The Joker single's songwriters are Eddie Curtis, Ahmet Ertegün, and Steve Miller.
  Fact 0.11: The Joker single's producer is Steve Miller.
  Fact 0.12: The Joker single is a cover of a 1973 single.
  Fact 0.13: Steve Miller Band singles chronology includes 'Fandango' (1972), 'The Joker' (1973), and 'Your Cash Ain't Nothin' But Trash' (1974).
  Fact 0.14: 'Take the Money and Run' / 'The Joker' (1983) was a double A-side live single by Steve Miller Band.
  Fact 0.15: 'The Joker' (1983) live single is a cover of the 1983 live single.
  Fact 0.16: Steve Miller Band singles chronology includes 'Living in the U.S.A.' (live) (1983).
  Fact 0.17: Steve Miller Band singles chronology includes 'Take the Money and Run' (live) / 'The Joker' (live) (1983).
  Fact 0.18: Steve Miller Band singles chronology includes 'Shangri-La' (1984).
  Fact 0.19: Steve Miller Band singles chronology includes 'The Joker' (1990, reissue), a cover of the 1990 reissue.
  Fact 0.20: Steve Miller Band singles chronology includes 'Ya Ya' (1988).
  Fact 0.21: Steve Miller Band singles chronology includes 'The Joker' (1990).
  Fact 0.22: Steve Miller Band singles chronology includes 'Wide River' (1993).
Sentence 1: `` The Joker '' is a song by the Steve Miller Band from their 1973 album The Joker . It is one of two Steve Miller Band songs that feature the nonce word `` pompatus '' . The song topped the US Billboard Hot 100 in early 1974 .
  Fact 1.0: 'The Joker' is a song by the Steve Miller Band.
  Fact 1.1: 'The Joker' is from the Steve Miller Band's 1973 album The Joker.
  Fact 1.2: 'The Joker' is one of two Steve Miller Band songs that feature the nonce word 'pompatus'.
  Fact 1.3: 'The Joker' topped the US Billboard Hot 100 in early 1974.

Retrieved items:
<item 1>
The Joker (Steve Miller Band song)
Luke Bryan often performs the song near the end of his concerts . Sampling ( edit ) `` The Joker '' was sampled in the song `` Summer and Lightning '' by
Electric Light Orchestra from their 1977 album , Out of the Blue . `` The Joker '' serves as the outline of the song `` So Cal Loco ( Party Like a Rockstar ) ''
by Sprung Monkey which appeared in the film Dude , Where 's My Car ? . It was also sampled on the original Def American Records release of the Geto Boys ' song
`` Gangster of Love '' from their 1989 album , Grip It ! On That Other Level . The sample , presumably unauthorized , was replaced with Lynyrd Skynyrd 's ``
Sweet Home Alabama '' on later pressings of the album . The song `` Hudson Dragonfly '' by Jackie - O Motherfucker , from the album , America Mystica ( 2006 ) ,
contains a sample of `` The Joker '' . The song `` Angel '' by Shaggy , from the album Hot Shot ( 2000 ) , samples the bass line of `` The Joker '' .
</item 1>

<item 2>
The Joker (Steve Miller Band song)
Formats and track listings ( edit ) 7 '' single ( 1973 ) `` The Joker '' -- 3 : 36 `` Something to Believe In '' -- 4 : 40 7 '' single ( 1983 - live version )
`` The Joker '' ( live ) -- 2 : 55 `` Take the Money and Run '' ( live ) -- 3 : 49 7 '' single ( 1990 ) `` The Joker '' ( single version ) -- 3 : 34 `` Do n't
Let Nobody Turn You Around '' -- 2 : 27 12 '' maxi ( 1990 ) `` The Joker '' ( LP version ) -- 4 : 22 `` Do n't Let Nobody Turn You Around '' -- 2 : 27 `` Shu Ba
Da Du Ma Ma Ma '' -- 5 : 39 CD maxi ( 1990 ) `` The Joker '' ( single version ) -- 3 : 34 `` Do n't Let Nobody Turn You Around '' -- 2 : 27 `` Shu Ba Da Du Ma
Ma Ma Ma '' -- 3 : 33 `` Living in the U.S.A. ''
</item 2>

<item 3>
The Joker (Steve Miller Band song)
Songwriter ( s ) Eddie Curtis , Ahmet Ertegün , Steve Miller Producer ( s ) Steve Miller Steve Miller Band singles chronology `` Fandango '' ( 1972 ) `` The
Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash '' ( 1974 ) `` Fandango '' ( 1972 ) `` The Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash ''
( 1974 ) `` Take the Money and Run '' / `` The Joker '' ( 1983 , double A-side , live ) Cover of the 1983 live single Steve Miller Band singles chronology ``
Living in the U.S.A. ''
</item 3>

<item 4>
The Joker (Steve Miller Band song)
Retrieved October 6 , 2009 . Jump up ^ `` The Irish Charts - search result '' . Retrieved May 16 , 2016 . Jump up ^ `` norwegiancharts.com - Steve Miller Band -
The Joker '' . Retrieved October 6 , 2009 . Jump up ^ `` swedishcharts.com - Steve Miller Band - The Joker '' . Retrieved October 6 , 2009 . Jump up ^ `` Steve
Miller Band - The Joker - hitparade.ch '' ( in German ) . Retrieved October 6 , 2009 . Jump up ^ `` End of Year Charts 1991 '' . Recorded Music NZ . Retrieved
December 3 , 2017 . Jump up ^ `` Chart Stats - Fatboy Slim - The Joker '' . Retrieved October 13 , 2009 . Jump up ^ `` Smashing Pumpkins dot com '' . Retrieved
September 28 , 2009 . Jump up ^ `` The Howard Stern Radio Show '' . New York . 17 Apr 2013 . Sirius XM Satellite Radio . Howard 100 .
</item 4>

<item 5>
The Joker (Steve Miller Band song)
Steve Miller recorded an unreleased demo of this song 's music with the lyrics of his other song , Take the Money and Run . The 2012 song Diese Tage with German
and English lyrics by KRIS feat Dante Thomas uses the chorus of `` The Joker '' . Country rapper Colt Ford sampled this song for his song `` Gangsta of Love ''
. `` Wow Wow '' by Neil Cicierega from the album `` Mouth Moods '' mixes `` The Joker '' with Will Smith 's `` Wild Wild West '' , `` All Star '' by Smash Mouth
and `` Cowboy '' by Kid Rock . In popular culture ( edit ) In 1996 , `` The Joker '' inspired a comedy film , The Pompatus of Love . References and footnotes (
edit ) ^ Jump up to : `` Steve Miller Band > Charts & Awards > Billboard Singles '' . allmusic . Retrieved October 5 , 2009 . ^ Jump up to : `` Chart Stats -
The Steve Miller Band - The Joker '' . Retrieved October 5 , 2009 . Jump up ^ `` irishcharts.ie search results '' .
</item 5>

<item 6>
The Joker (Steve Miller Band song)
-- 3 : 59 Personnel ( edit ) Steve Miller -- guitar , lead vocals Gerald Johnson -- bass , backing vocals Dick Thompson -- organ John King -- drums
Certifications ( edit ) Country Certification Date Sales certified Sweden Gold February 6 , 1991 25,000 United Kingdom Silver October 1 , 1990 200,000 United
States Gold January 11 , 1974 1,000,000 Charts ( edit ) Weekly charts ( edit ) Chart ( 1973 / 1974 ) Peak position US Billboard Hot 100 Canadian RPM Top Singles
Dutch Daverende 30 18 Dutch Top 40 18 Australia Kent Music Report 8 Chart ( 1990 / 1991 ) Peak position Austrian Singles Chart 5 Dutch Nationale Top 100 Dutch
Top 40 French SNEP Singles Chart 33 Irish Singles Chart New Zealand RIANZ Singles Chart Norwegian Singles Chart Swedish Singles Chart Swiss Singles Chart 5 UK
Singles Chart Year - end charts ( edit ) Chart ( 1991 ) Position New Zealand ( Recorded Music NZ ) 20 Preceded by `` Time in a Bottle '' by Jim Croce Billboard
Hot 100 number one single January 12 ,
</item 6>

<item 7>
The Joker (Steve Miller Band song)
The Joker ( Steve Miller Band song ) - wikipedia The Joker ( Steve Miller Band song ) Jump to : navigation , search `` The Joker ( song ) '' redirects here .
For other songs with the same title , see Joker ( disambiguation ) . `` The Joker '' Cover of the 1973 single Single by Steve Miller Band from the album The
Joker B - side `` ' see track listings ' '' Released October 1973 Format 7 '' Single , 12 '' single , Cassette single ( Original Releases ) , CD single and
Digital download ( Post Releases ) . Genre Rock , blues rock , southern rock Length 4 : 26 ( Album version ) ; 3 : 35 ( 7 '' version / Radio edit ) and 6 : 20 (
Extended version ) Label Capitol Records ; EMI - Odeon ; London Records , Parlophone Records , EMI Bovema , Odeon Records .
</item 7>

<item 8>
The Joker (Steve Miller Band song)
The song topped the US Billboard Hot 100 in early 1974 . More than 16 years later , in September 1990 , it reached number one in the UK Singles Chart for two
weeks after being used in `` Great Deal '' , a Hugh Johnson - directed television advertisement for Levi 's , thus holding the record for the longest gap
between transatlantic chart - toppers . This reissue of `` The Joker '' also topped the Irish Singles Chart , the New Zealand RIANZ Singles Chart , the Dutch
Nationale Top 100 and the Dutch Top 40 . The first line of the lyrics is a reference to the song `` Space Cowboy '' from Miller 's Brave New World album .
Following lines refer to two other songs : `` Gangster of Love '' from Sailor and `` Enter Maurice '' from Recall the Beginning ... A Journey from Eden .
</item 8>

<item 9>
The Joker (Steve Miller Band song)
lang , Craig Street k.d. lang singles chronology `` Sexuality '' ( 1996 ) `` The Joker '' ( 1997 ) `` Theme from the Valley of the Dolls '' ( 1997 ) ``
Sexuality '' ( 1996 ) `` The Joker '' ( 1997 ) `` Theme from the Valley of the Dolls '' ( 1997 ) `` The Joker '' Single by Fatboy Slim from the album
Palookaville Released February 28 , 2005 ( 2005 - 02 - 28 ) Recorded Genre Big beat Label Astralwerks , Skint Songwriter ( s ) Eddie Curtis , Ahmet Ertegün ,
Steve Miller Producer ( s ) Fatboy Slim , Simon Thornton Fatboy Slim singles chronology `` Wonderful Night '' ( 2004 ) `` The Joker '' ( 2005 ) `` Do n't Let
the Man Get You Down '' ( 2005 ) `` Wonderful Night '' ( 2004 ) `` The Joker '' ( 2005 ) `` Do n't Let the Man Get You Down '' ( 2005 ) Alternative cover Remix
single Covers ( edit ) k.d.
</item 9>

<item 10>
The Joker (Steve Miller Band song)
Lyrics of this song at MetroLyrics Steve Miller Band Steve Miller Kenny Lee Lewis Gordy Knudtson Joseph Wooten Jacob Peterson Norton Buffalo Tim Davis Boz
Scaggs Lonnie Turner James Cooke Lance Haas Jim Peterman Glyn Johns Ben Sidran Nicky Hopkins Bobby Winkelman Charlie McCoy Jack King Ross Valory Gerald Johnson
Dick Thompson Jim Keltner Roger Allen Clark Gary Mallaber Nick DeCaro John King David Denny Greg Douglass Byron Allred John Massaro Billy Peterson Ricky
Peterson Bob Malach Leo Sidran Jay Bird Koder Joey Heinemann Sonny Charles Studio albums Children of the Future Sailor Brave New World Your Saving Grace Number
5 Rock Love Recall the Beginning ... A Journey from Eden The Joker Fly Like an Eagle Book of Dreams Circle of Love Abracadabra Italian X Rays Living in the 20th
Century Born 2 B Blue ( Steve Miller solo album ) Wide River Bingo ! Let Your Hair Down Live albums Live at the Fillmore Auditorium Steve Miller Band Live !
Compilations Anthology Living in the U.S.A.
</item 10>
=== END ITEM id=57ea7058b182e4ab ===

=== ITEM id=2865ca165949faad ===
Gold sentences and facts:
Sentence 0: `` The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit . In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
  Fact 0.0: The Glory of Love is a song.
  Fact 0.1: The Glory of Love was written by Billy Hill.
  Fact 0.2: The Glory of Love was recorded by Benny Goodman in 1936.
  Fact 0.3: Benny Goodman's version of The Glory of Love was a number one pop hit.
  Fact 0.4: In 1951, the R&B vocal group The Five Keys had their biggest R&B hit with their version of The Glory of Love.
  Fact 0.5: The Five Keys' version hit number one on the R&B chart for four non-consecutive weeks.
  Fact 0.6: The Five Keys' recording reportedly sold a million copies.
  Fact 0.7: Pressed recordings of The Five Keys' version are very rare.
Sentence 1: `` The Glory of Love '' Single by Benny Goodman Released 1936 Format 78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952
  Fact 1.0: The Glory of Love is a single by Benny Goodman.
  Fact 1.1: The Glory of Love was released in 1936.
  Fact 1.2: The Glory of Love has the format 78 rpm vinyl.
  Fact 1.3: The Glory of Love was recorded in 1936.
  Fact 1.4: The Glory of Love has the genre pop.
  Fact 1.5: The songwriter of The Glory of Love is Billy Hill.
  Fact 1.6: The Benny Goodman singles chronology lists The Glory of Love (1936).
  Fact 1.7: The Five Keys singles chronology lists The Glory of Love (1951).
  Fact 1.8: The Five Keys singles chronology lists Yes Sir, That's My Baby (1952).

Retrieved items:
<item 1>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 1>

<item 2>
God Save the Queen
Some of these are : The French Marquise de Créquy wrote in her Souvenirs that the words `` Grand Dieu , sauvez le Roi ! '' were indeed composed by Jean -
Baptiste Lully in gratitude for the survival by Louis XIV of an anal fistula operation . ( The surgical knife purpose - made for the occasion is on display in
the Musée d'histoire de la médecine . ) Allegedly , Lully had set words by Marie de Brinon to music , and Créquy claimed that the tune was later plagiarised by
Handel , and so adopted by the British as `` un de leurs air nationaux '' . ( The entire story might have been intended as a joke . ) Translated into Latin
under the title Domine , Salvum Fac Regem , it became the French anthem until 1792 . After the Battle of Culloden , the Hanover dynasty supposedly adopted this
melody as the British anthem . James Oswald was a possible author of the Thesaurus Musicus , so may have played a part in the history of the song , but is not a
strong enough candidate to be cited as the composer of the tune .
</item 2>

<item 3>
Let My Love Open the Door
Let My Love Open the Door - wikipedia Let My Love Open the Door Jump to : navigation , search `` Let My Love Open the Door '' Single by Pete Townshend from the
album Empty Glass B - side `` Greyhound Girl '' Released June 4 , 1980 ( US ) Format 7 '' Recorded March 1980 Genre Art rock , new wave Length 2 : 44 Label Atco
Records Songwriter ( s ) Pete Townshend Producer ( s ) Pete Townshend ,
</item 3>

<item 4>
“The Glory of Love,” written by Billy Hill, was a 1936 Benny Goodman pop hit. The Five Keys’ 1951 version topped the R&B chart for four nonconsecutive weeks;
Otis Redding’s 1967 cover reached No. 19 R&B and No. 60 on the Hot 100.

Pete Townshend wrote and recorded “Let My Love Open the Door” for his 1980 album *Empty Glass*. It peaked at No. 9
</item 4>

<item 5>
The Glory of Love (song)
Redding 's cover became a top 20 hit and reached number nineteen on the Billboard R&B Songs chart and number sixty on the Billboard Hot 100 . Charts ( edit )
Weekly charts ( edit ) Chart ( 1967 ) Peak position US Billboard Hot 100 60 US Billboard Hot Rhythm & Blues Songs 19 References ( edit ) Jump up ^ Whitburn ,
Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record Research . p. 205 . Jump up ^ Propes , Steve ( 1973 ) . Those Oldies But Goodies : A Guide to
50 's Record Collecting . The Macmillan Company , New York . p. 42 . Jump up ^ Whitburn , Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record
Research . p. 486 . Jump up ^ `` allmusic ( ( ( Otis Redding > Awards ) ) ) '' . Billboard . Retrieved 2012 - 11 - 22 .
</item 5>

<item 6>
The Glory of Love (song)
The Glory of Love ( song ) - wikipedia The Glory of Love ( song ) Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2015 ) ( Learn how and
when to remove this template message ) For the song by Peter Cetera , see Glory of Love . `` The Glory of Love '' Single by Benny Goodman Released 1936 Format
78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The
Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952 ``
The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit .
</item 6>

<item 7>
The Glory of Love (song)
Although The Five Keys recording sold a reported million copies pressed recordings are very rare . Contents ( hide ) 1 Charts 1.1 Weekly charts 2 Otis Redding
version 3 Charts 3.1 Weekly charts 4 References Charts ( edit ) Weekly charts ( edit ) Chart ( 1951 ) Peak position US Billboard Hot Rhythm & Blues Songs Otis
Redding version ( edit ) `` The Glory of Love '' Single by Otis Redding from the album The Dock of the Bay B - side `` I 'm Coming Home '' Released 1967 Format
7 '' vinyl Recorded December , 1967 Genre Soul , Pop Length 2 : 38 Label Volt S419 Songwriter ( s ) Billy Hill Producer ( s ) Steve Cropper Otis Redding singles
chronology `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 ) `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 )
In 1967 , Otis Redding recorded a cover version for his 1968 album , The Dock of the Bay .
</item 7>

<item 8>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 8>

<item 9>
The Glory of Love (song)
In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four
non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
</item 9>

<item 10>
The Glory of Love (song)
Billboard . Retrieved 2012 - 11 - 22 . Jump up ^ `` Billboard R&B Singles Chart '' . Billboard . Retrieved 2016 - 01 - 27 . Preceded by `` Sixty Minute Man ''
by The Dominoes Billboard Best Selling Retail Rhythm & Blues Records number - one single ( The Five Keys version ) September 22 , 1951 Succeeded by `` Sixty
Minute Man '' by The Dominoes Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Glory_of_Love_(song)&oldid=814898703 '' Categories : 1936 songs
1951 singles Benny Goodman songs Otis Redding songs Songs written by Billy Hill ( songwriter ) Billboard Hot R&B / Hip - Hop Songs number - one singles Paul
McCartney songs Hidden categories : Articles needing additional references from December 2015 All articles needing additional references Music infoboxes with
deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia Norsk Edit links This page was last edited on 11 December 2017 , at 15 :
51 . About Wikipedia
</item 10>
=== END ITEM id=2865ca165949faad ===

=== ITEM id=3aeaf7a19fe2ceab ===
Gold sentences and facts:
Sentence 0: Nintendo claims that the design of the Pro Controller is an `` enhanced version '' of the Classic Controller and `` offers a richer experience '' . Certain games with flexible control schemes , such as Call of Duty : Black Ops II and Trine 2 : Director 's Cut , are also compatible with the Classic Controller . The Wii U Pro Controller is not compatible with previous - generation Wii games .
  Fact 0.0: Nintendo claims that the design of the Wii U Pro Controller is an enhanced version of the Classic Controller.
  Fact 0.1: Nintendo claims that the Wii U Pro Controller offers a richer experience.
  Fact 0.2: Certain games with flexible control schemes are compatible with the Classic Controller.
  Fact 0.3: Call of Duty: Black Ops II is a game with a flexible control scheme that is compatible with the Classic Controller.
  Fact 0.4: Trine 2: Director's Cut is a game with a flexible control scheme that is compatible with the Classic Controller.
  Fact 0.5: The Wii U Pro Controller is not compatible with previous-generation Wii games.
Sentence 1: The Wii U Pro Controller ( Wii U PRO コントローラー ) is a video game controller produced by Nintendo for the Wii U video game console .
  Fact 1.0: The Wii U Pro Controller is a video game controller.
  Fact 1.1: The Wii U Pro Controller is produced by Nintendo.
  Fact 1.2: The Wii U Pro Controller is made for the Wii U video game console.
Sentence 2: The controller functions as a secondary controller released for the console , available separately . The Wii U system can be connected to up to four Wii U Pro Controllers at one time , however Super Smash Bros. for Wii U allows the usage of more than 4 . Like more traditional controllers , it features standard analog sticks and face buttons . Like the Wii U GamePad and Classic Controller Pro and unlike other eighth generation controllers ( i.e. , the DualShock 4 and the Xbox One Controller ) , the triggers are digital ( i.e. , not analog ) .
  Fact 2.0: The Wii U Pro Controller functions as a secondary controller for the Wii U.
  Fact 2.1: The Wii U Pro Controller was released for the console.
  Fact 2.2: The Wii U Pro Controller is available separately.
  Fact 2.3: The Wii U console can be connected to up to four Wii U Pro Controllers at one time.
  Fact 2.4: Super Smash Bros. for Wii U allows the use of more than four controllers.
  Fact 2.5: The Wii U Pro Controller has standard analog sticks.
  Fact 2.6: The Wii U Pro Controller has face buttons, like more traditional controllers.
  Fact 2.7: The Wii U Pro Controller has digital triggers, not analog triggers.
  Fact 2.8: The Wii U GamePad has digital triggers.
  Fact 2.9: The Classic Controller Pro has digital triggers.
  Fact 2.10: Unlike the DualShock 4 and the Xbox One Controller, the Wii U Pro Controller has digital triggers.

Retrieved items:
<item 1>
History of Nintendo
Law360.com . Retrieved 17 December 2014 . Jump up ^ `` Wii European launch details announced '' . GamesIndustry.biz. 15 September 2006 . Retrieved 18 December
2014 . Jump up ^ Internet Archive Wayback Machine Jump up ^ Gaudiosi , John ( 25 April 2007 ) . `` The untold story of how the Wii beat the Xbox , PlayStation
'' . CNN.com . Retrieved 18 December 2014 . Jump up ^ Stan Schroeder , Mashable ( 25 April 2011 ) . `` Nintendo confirms Wii successor for 2012 '' . CNN.com .
Retrieved 2012 - 10 - 31 . Jump up ^ `` Wii U confirmed for Europe this year '' . Metro . Jump up ^ `` WiiU Price and Release Date Announced '' . WiiU News .
2012 - 09 - 13 . Retrieved 2012 - 09 - 13 . Jump up ^ `` Nintendo Network @ E3 2011 - Introducing Wii U '' . Web.archive.org. 2011 - 06 - 15 .
</item 1>

<item 2>
Wii U Pro Controller
Wii U Pro Controller - wikipedia Wii U Pro Controller Jump to : navigation , search Wii U Pro Controller Wii U Pro Controller Manufacturer Nintendo Type Video
game controller Generation Eighth generation Retail availability November 18 , 2012 Input 2 × analog sticks Digital d - pad 10 × digital face buttons ( two
buttons found under analog sticks ; controller features additional button for syncing purposes ) 2 × digital shoulder buttons 2 × digital triggers Connectivity
Bluetooth Power 1300 mAh , Mini USB B connector ( recharge ) Predecessor Wii Remote ( Plus ) and Classic Controller ( Pro ) Successor Nintendo Switch Pro
Controller The Wii U Pro Controller ( Wii U PRO コントローラー ) is a video game controller produced by Nintendo for the Wii U video game console . History ( edit )
Nintendo unveiled the Wii U Pro Controller at E3 2012 ; the controller is aimed towards attracting `` hardcore '' players to the platform with a more
traditional design , to make the Wii U more competitive with Sony 's and Microsoft 's offerings . Many video game journalists have noted the similarity between
the controller and Microsoft 's Xbox 360 Controller .
</item 2>

<item 3>
History of Nintendo
Nintendo 3DS , Wii U , and Nintendo Switch ( edit ) Main articles : Nintendo 3DS , Nintendo 3DS XL , Nintendo 2DS , New Nintendo 3DS , Wii U , and Nintendo
Switch The Wii U console with Wii U GamePad On February 26 , 2011 , the Nintendo 3DS was released in Japan . It was released on March 25 and 27 to Europe and
the US , respectively . The 3DS was part of a large wave of 3D technologies being released at the time ; unlike most of these , the 3DS did not use 3D glasses
for its 3D effect , and instead used parallax barrier autostereoscopy . On April 25 , 2011 , Nintendo confirmed that they were making a new console to be
released in 2012 . On June 7 , 2011 , the console was announced as being named the Wii U . The Wii U is Nintendo 's first console to be able to play video games
in 1080p HD .
</item 3>

<item 4>
Wii U Pro Controller
Features ( edit ) See also : List of Wii U games that use the Wii U Pro Controller The controller functions as a secondary controller released for the console ,
available separately . The Wii U system can be connected to up to four Wii U Pro Controllers at one time , however Super Smash Bros. for Wii U allows the usage
of more than 4 . Like more traditional controllers , it features standard analog sticks and face buttons . Like the Wii U GamePad and Classic Controller Pro and
unlike other eighth generation controllers ( i.e. , the DualShock 4 and the Xbox One Controller ) , the triggers are digital ( i.e. , not analog ) . The Wii U
Pro Controller uses the same 1300mAh CTR - 003 battery found in the Nintendo 3DS and 2DS , which can last up to 80 hours before needing to be recharged .
Nintendo claims that the design of the Pro Controller is an `` enhanced version '' of the Classic Controller and `` offers a richer experience '' . Certain
games with flexible control schemes , such as Call of Duty : Black Ops II and Trine 2 : Director 's Cut , are also compatible with the Classic Controller .
</item 4>

<item 5>
History of Nintendo
During the Electronics Entertainment Expo in 2011 , Nintendo 's press conference put a great amount of stress on the abilities and innovations of the new
controller , a unit resembling a tablet . Many gamers were confused after the press conference about whether the Wii U is an entirely new system or just a new
controller for the Wii . The Wii U was , however , a whole new home console system developed by Nintendo . The system was released on November 18 , 2012 in
North America , November 30 , 2012 in Europe and Australia , and on December 8 , 2012 in Japan . It featured a new controller with a 6.2 inch touchscreen , two
thumb pads , four shoulder buttons , a directional pad , and four lettered buttons . It also featured an accelerometer , gyroscope , camera , Near - field
communications ( NFC ) , and built - in microphone , similar those in the Nintendo 3DS . The Nintendo Switch Console in the Dock with Red and Blue Joy - Con
controllers . On July 11 , 2015 , Satoru Iwata died from a bile duct tumor at the age of 55 .
</item 5>

<item 6>
History of Nintendo
It was launched in Japan on March 2 , 2006 , and three months later in North America and Europe on June 11 and June 23 , 2006 , respectively . On May 25 , 2006
, Reggie Fils - Aimé was promoted to President and COO of Nintendo of America , Inc . The former President , Tatsumi Kimishima , was promoted to Chairman of the
Board and CEO . On July 7 , 2006 , Nintendo officially established a South Korean subsidiary , Nintendo Korea , in the country 's capital , Seoul , which
replaced Daewon Media as the official distributor of Nintendo products in South Korea . Original Wii console with Wii Remote In early August 2006 , it was
revealed that Nintendo , along with Microsoft , was made the target of a patent - infringement lawsuit . Leveled by the Anascape Ltd. , the suit claimed that
Nintendo 's use of analog technology in their game controllers constitutes a violation of their patents . The lawsuit sought to recover damages from both
corporations and possibly force them to stop selling controllers with the violating technology .
</item 6>

<item 7>
Wii U Pro Controller
wikipedia.org/w/index.php?title=Wii_U_Pro_Controller&oldid=837880962 '' Categories : Nintendo controllers Wii U Hidden categories : Pages using deprecated image
syntax Articles containing Japanese - language text Talk About Wikipedia Deutsch Español Italiano Português Edit links This page was last edited on 23 April
2018 , at 16 : 07 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree
to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Wii U Pro Controller
The Wii U Pro Controller is not compatible with previous - generation Wii games . References ( edit ) Jump up ^ Franich , Darren ( June 3 , 2012 ) . `` The Wii
U Pro Controller looks very familiar ... '' EW.com . Retrieved July 28 , 2012 . Jump up ^ Kain , Erik ( June 5 , 2012 ) . `` Nintendo 's Smart Decision With Its
Wii U Pro Controller '' . Forbes . Retrieved July 28 , 2012 . Jump up ^ Grubb , Jeffrey ( June 3 , 2012 ) . `` Nintendo 's Wii U Pro Controller may make
Microsoft blush '' . VentureBeat . Retrieved July 28 , 2012 . Jump up ^ How to Sync a Wii Remote with a Wii U Console Jump up ^ `` Wii U Pro Controller and 3DS
share the same battery '' . gonintendo.com . November 9 , 2012 . Retrieved August 16 , 2014 . Jump up ^ `` We Opened Up The Nintendo 2DS And Found A 1300mAh
Battery Inside '' .
</item 8>

<item 9>
Wii U Pro Controller
Tomshardware.com . Retrieved November 26 , 2012 . Wii U Hardware Wii U GamePad Off - TV Play Wii U Pro Controller Wii accessories Wii Remote ( Plus ) MotionPlus
Nunchuk Classic Controller Balance Board Wii Speak Third - party accessories Amiibo Online service Nintendo Network Miiverse Nintendo eShop Nintendo TVii Amazon
Instant Video / LoveFilm Hulu Plus Netflix TiVo SpotPass System software Home Menu Amazon Instant Video LoveFilm Crunchyroll Hulu Napster Netflix Niconico Mii
Maker YouTube Wii Mode Wii U Chat Games List Best - selling Balance Board Pro Controller Miis Nintendo Network Wii MotionPlus Virtual Console Japan North
America PAL region Backward compatibility Wii games Wi - Fi Connection Wii applications WiiWare games North America PAL region Wii Mode Virtual Console Japan
North America PAL region Wii series Street U Fit U Party U Sports Club Miscellaneous List of Wii games on Wii U eShop Licensed toolkits Integrated Nintendo Web
Framework Unity Supported Autodesk Gameware CryEngine ( 4th generation ) Havok Gamebryo Umbra Unreal Engine 3 ← Predecessor : Wii Follow - up : Nintendo Switch
→ Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
Wii U Pro Controller
nintendolife.com . October 3 , 2013 . Retrieved August 16 , 2014 . Jump up ^ `` Wii U Pro Controller Boasts 80 - Hour Battery Life - IGN '' . Uk.ign.com .
October 16 , 2012 . Retrieved September 21 , 2013 . Jump up ^ Liebl , Matt ( June 12 , 2012 ) . `` Nintendo defends Wii U Pro Controller design '' . GameZone .
Retrieved July 28 , 2012 . Jump up ^ `` Wii U Pro Controller Not Compatible With Wii Games '' . Tomshardware.com . Retrieved November 26 , 2012 .
</item 10>
=== END ITEM id=3aeaf7a19fe2ceab ===

=== ITEM id=f62927b3b4282f44 ===
Gold sentences and facts:
Sentence 0: The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes ) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman , Jr . Irving Lippman Camera setup Single - camera Running time 25 minutes Production company ( s ) Screen Gems Television ( 1970 -- 1974 ) Distributor Columbia Pictures Television ( 1974 -- 84 ; 1989 -- 96 ) DFS Program Exchange ( 1984 -- 87 ) The Program Exchange ( 1987 -- 89 ) Columbia TriStar Television ( 1996 -- 2002 ) Sony Pictures Television ( 2002 -- present ) Release Original network ABC Audio format Monaural Original release September 25 , 1970 ( 1970 - 09 - 25 ) -- March 23 , 1974 ( 1974 - 03 - 23 ) Chronology Related shows Getting Together Goober and the Ghost Chasers Partridge Family 2200 A.D.
  Fact 0.0: The Partridge Family is a musical sitcom
  Fact 0.1: The Partridge Family was created by Bernard Slade
  Fact 0.2: The Partridge Family starred Shirley Jones
  Fact 0.3: The Partridge Family starred David Cassidy
  Fact 0.4: The Partridge Family starred Danny Bonaduce
  Fact 0.5: The Partridge Family starred Susan Dey
  Fact 0.6: The Partridge Family starred Suzanne Crough
  Fact 0.7: The Partridge Family starred Jeremy Gelbwaks
  Fact 0.8: The Partridge Family starred Dave Madden
  Fact 0.9: The Partridge Family starred Brian Forster
  Fact 0.10: The Partridge Family's theme music composers were Diane Hilderbrand, Danny Janssen and Wes Farrell
  Fact 0.11: The Partridge Family's opening theme was 'When We're Singin'' (1970-71)
  Fact 0.12: The Partridge Family's opening theme was 'C'mon, Get Happy' (1971-74)
  Fact 0.13: The Partridge Family's composers were George Duning, Benny Golson, Warren Barker, Hugo Montenegro and Shorty Rogers
  Fact 0.14: The Partridge Family is from the United States
  Fact 0.15: The Partridge Family's original language is English
  Fact 0.16: The Partridge Family has 96 episodes
  Fact 0.17: The Partridge Family's executive producer was Bob Claver
  Fact 0.18: The Partridge Family's producers were William Bickley, Paul Junger Witt, Dale McRaven, Larry Rosen and Mel Swope
  Fact 0.19: The Partridge Family's cinematography was by Fred Jackman, Jr. and Irving Lippman
  Fact 0.20: The Partridge Family uses a single-camera setup
  Fact 0.21: The Partridge Family has a running time of 25 minutes
  Fact 0.22: The Partridge Family was produced by Screen Gems Television (1970-1974)
  Fact 0.23: The Partridge Family was distributed by Columbia Pictures Television (1974-84; 1989-96)
  Fact 0.24: The Partridge Family was distributed by DFS Program Exchange (1984-87)
  Fact 0.25: The Partridge Family was distributed by The Program Exchange (1987-89)
  Fact 0.26: The Partridge Family was distributed by Columbia TriStar Television (1996-2002)
  Fact 0.27: The Partridge Family was distributed by Sony Pictures Television (2002-present)
  Fact 0.28: The Partridge Family aired originally on ABC
  Fact 0.29: The Partridge Family has a monaural audio format
  Fact 0.30: The Partridge Family originally aired from September 25, 1970 to March 23, 1974
  Fact 0.31: Related shows include Getting Together
  Fact 0.32: Related shows include Goober and the Ghost Chasers
  Fact 0.33: Related shows include Partridge Family 2200 A.D.

Retrieved items:
<item 1>
The Partridge Family
Actor David Cassidy sang lead on most of the records , and actor Shirley Jones can be heard on a few . But the other voices and all the instruments were
recorded by session musicians . The Partridge Family 's instrumental sound was provided by musicians informally known as The Wrecking Crew , who did many of the
pop hits of the day attributed to other performers . The actors pretended to play their instruments . None of the actors sang background ; they lip - synched to
recorded singing by The Love Generation , AKA The Ron Hicklin Singers , composed of brothers John and Tom Bahler , Jackie Ward ( who as Robin Ward had a # 14
hit `` Wonderful Summer '' on Dot Records in 1963 ) , and Ron Hicklin . Actor David Cassidy was originally to lip sync with the rest of the cast , but convinced
Farrell that he could sing and was allowed to join the studio ensemble as the lead singer . He and his step - mother , Shirley , are the only two show actors
allowed to sing for recording . The Partridge Family theme song , produced by Rogers , was shown over opening credits . More than one song was used as the theme
song .
</item 1>

<item 2>
The Partridge Family
Episodes ( edit ) Main article : List of The Partridge Family episodes Shirley Jones and Ricky Segall , season 4 Production ( edit ) At the end of the first
season , Jeremy Gelbwaks ' family moved out of the Los Angeles area , and the part of Chris was recast with actor Brian Forster . According to David Cassidy ,
Gelbwaks `` had a personality conflict with every person in the cast and the producers '' and especially did not get along with Cassidy or Bonaduce . A dog
named Simone was featured in the first season , but it was phased out during the second season . At the beginning of the fourth season , a four - year - old
neighbor , Ricky Stevens ( Ricky Segall ) , was featured and would sing a children 's song during each episode , but the character was dropped mid-season .
Music ( edit ) Music recorded for the pilot episode was produced by Monkees ' producer Shorty Rogers . Songs for the ongoing series were recorded by music
producer Wes Farrell . The Partridge Family songs heard in the show , and on the several Partridge Family singles and albums , were not performed as seen on -
screen .
</item 2>

<item 3>
The Partridge Family
1 hit ( after The Chipmunks and The Archies ) . The song 's companion LP , The Partridge Family Album , reached Number 4 in the Billboard 200 . It was also
awarded a gold disc by the RIAA in December 1970 , having sold over 500,000 copies . A string of hit Partridge singles followed : `` Does n't Somebody Want to
Be Wanted '' , `` I 'll Meet You Halfway '' , `` I Woke Up In Love This Morning '' , `` It 's One of Those Nights ( Yes Love ) '' , `` Am I Losing You '' , ``
Looking Through the Eyes of Love '' , `` Breaking Up Is Hard to Do '' , and `` A Friend and a Lover '' . These singles were showcased on million - selling
albums including Up To Date , Sound Magazine , Shopping Bag , Notebook , Crossword Puzzle , and Bulletin Board . Their holiday album A Partridge Family
Christmas Card was the No. 1 selling Christmas record of 1971 .
</item 3>

<item 4>
The Partridge Family
The company promoted the show by releasing a series of albums featuring the family band , though David Cassidy and Shirley Jones , who sang background , were
the only cast members who were actually featured on the recordings . As the show and other associated merchandising soared , David Cassidy became a teen idol .
The producers signed Cassidy as a solo act as well . Cassidy began touring with his own group of musicians , performing Partridge songs as well as hits from his
own albums , to thousands of screaming teenagers in major stadiums across the USA , UK , Europe , Japan and Australia . The Partridge Family 's biggest hit came
in 1970 with the song `` I Think I Love You '' , written by Tony Romeo ( who had previously written several of the Cowsills ' hits ) , peaked at Number 1 on the
Billboard charts in December of that year . It sold over five million copies , outselling the Beatles ' `` Let It Be '' , was awarded a gold disc , and made the
group the third fictional artist to have a No. 1 hit ( after The Chipmunks and The Archies ) .
</item 4>

<item 5>
The Partridge Family
Jump up ^ `` TSORT Song Artist 592 -- The Partridge Family '' . TSORT.info . Retrieved June 22 , 2010 . Jump up ^ `` TSORT Album Artist 994 -- The Partridge
Family '' . TSORT.info . Retrieved June 22 , 2010 . Jump up ^ Whitburn , Joel ( 2004 ) . Christmas in the Charts ( 1920 -- 2004 ) . Wisconsin : Record Research
Inc. p. 179 . ISBN 0 - 89820 - 161 - 6 . Jump up ^ Parry , Wayne ( April 10 , 2011 ) . David Cassidy , Danny Bonaduce play Partridge song . Associated Press .
Retrieved 2011 - 04 - 10 . Jump up ^ Coopee , Todd . `` The Partridge Family Game '' . ToyTales.ca . Jump up ^ Shirley , Ian ( 2005 ) . Can Rock & Roll Save the
World ? : An Illustrated History of Music and Comics . SAF Publishing Ltd. p. 88 - 89 .
</item 5>

<item 6>
The Partridge Family
Irving Lippman Camera setup Single - camera Running time 25 minutes Production company ( s ) Screen Gems Television ( 1970 -- 1974 ) Distributor Columbia
Pictures Television ( 1974 -- 84 ; 1989 -- 96 ) DFS Program Exchange ( 1984 -- 87 ) The Program Exchange ( 1987 -- 89 ) Columbia TriStar Television ( 1996 --
2002 ) Sony Pictures Television ( 2002 -- present ) Release Original network ABC Audio format Monaural Original release September 25 , 1970 ( 1970 - 09 - 25 )
-- March 23 , 1974 ( 1974 - 03 - 23 ) Chronology Related shows Getting Together Goober and the Ghost Chasers Partridge Family 2200 A.D. The Partridge Family is
an American musical sitcom starring Shirley Jones and featuring David Cassidy . Jones played a widowed mother , and Cassidy played the oldest of her five
children who embarked on a music career .
</item 6>

<item 7>
The Partridge Family
The Partridge Family - wikipedia The Partridge Family Jump to : navigation , search The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring
Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny
Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson
Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes
) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman
, Jr .
</item 7>

<item 8>
The Partridge Family
( hide ) The Partridge Family David Cassidy Shirley Jones Danny Bonaduce Susan Dey Jeremy Gelbwaks Brian Forster Suzanne Crough Dave Madden Studio albums The
Partridge Family Album Up to Date Sound Magazine A Partridge Family Christmas Card Shopping Bag The Partridge Family Notebook Crossword Puzzle Bulletin Board
Other albums At Home with Their Greatest Hits The World of the Partridge Family Greatest Hits The Definitive Collection Come On Get Happy ! : The Very Best of
The Partridge Family Singles `` I Think I Love You '' `` Does n't Somebody Want to Be Wanted '' `` I 'll Meet You Halfway '' `` I Woke Up in Love This Morning
'' `` It 's One of Those Nights ( Yes Love ) '' `` Breaking Up Is Hard to Do '' `` Looking Through the Eyes of Love '' Related articles List of The Partridge
Family episodes The Partridge Family discography David Cassidy discography Wes Farrell Goober and the Ghost Chasers Partridge Family 2200 A.D. MusicBrainz :
3376b487 - 95b4 - 42e6 - b7f4 - 71cffffcad97 Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
The Partridge Family
The shows often contrast suburban life with the adventures of a show - business family on the road . After the first season , more of the show 's action takes
place in their hometown than on tour . History ( edit ) This section needs additional citations for verification . Please help improve this article by adding
citations to reliable sources . Unsourced material may be challenged and removed . ( May 2016 ) ( Learn how and when to remove this template message ) Created
for television by Bernard Slade , the series ' executive producer was Bob Claver . The show was inspired by and loosely based on The Cowsills , a family pop
music group that was famous in the late 1960s . In the show 's early development , the Cowsill children were considered by the producers , but because the
Cowsills were not trained actors and were too old for the roles as scripted , Slade and Claver abandoned that idea . Shirley Jones had already been signed as
mother Shirley Partridge and star of the show . The pilot was filmed in December 1969 . This unaired pilot differs from the pilot broadcast in 1970 .
</item 9>

<item 10>
The Partridge Family
More than one song was used as the theme song . Season 1 episodes feature the song `` When We 're Singin ' '' , which was replaced in subsequent seasons with ``
C'mon Get Happy '' . `` Five of us , and Mom working all day , we knew we could help her if our music would pay , Danny got Reuben to sell our song , and it
really came together when Mom sang along ... '' ( from `` When We 're Singin ' '' ) Later , under Farrell 's direction , the new version had new lyrics sung to
the `` When We 're Singin ' '' tune . With the new chorus finalized , `` C'mon Get Happy '' showcased the new verse : `` We had a dream , we 'd go travelin '
together , We 'd spread a little lovin ' then we 'd keep movin ' on .
</item 10>
=== END ITEM id=f62927b3b4282f44 ===

=== ITEM id=8890487e8d2009d2 ===
Gold sentences and facts:
Sentence 0: In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers . When Maximoff throws the explosion into the sky with telekinesis , it damages a nearby building , killing several Wakandan humanitarian workers .
  Fact 0.0: In 1991, the brainwashed super-soldier James 'Bucky' Barnes was dispatched from a Hydra base in Siberia.
  Fact 0.1: In 1991, James 'Bucky' Barnes was dispatched to intercept an automobile carrying a case of super-soldier serum.
  Fact 0.2: In the present day, Ultron's defeat in the nation of Sokovia at the hands of the Avengers happened approximately one year earlier.
  Fact 0.3: Ultron was defeated in the nation of Sokovia by the Avengers.
  Fact 0.4: In the present day, Steve Rogers, Natasha Romanoff, Sam Wilson, and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos.
  Fact 0.5: Brock Rumlow blows himself up.
  Fact 0.6: Brock Rumlow hoped to kill Steve Rogers by blowing himself up.
  Fact 0.7: Wanda Maximoff throws the explosion into the sky with telekinesis.
  Fact 0.8: The explosion damages a nearby building.
  Fact 0.9: The explosion killed several Wakandan humanitarian workers.

Retrieved items:
<item 1>
Captain America: Civil War
It is the sequel to 2011 's Captain America : The First Avenger and 2014 's Captain America : The Winter Soldier , and the thirteenth film in the Marvel
Cinematic Universe ( MCU ) . The film is directed by Anthony and Joe Russo , with a screenplay by Christopher Markus & Stephen McFeely , and features an
ensemble cast , including Chris Evans , Robert Downey Jr. , Scarlett Johansson , Sebastian Stan , Anthony Mackie , Don Cheadle , Jeremy Renner , Chadwick
Boseman , Paul Bettany , Elizabeth Olsen , Paul Rudd , Emily VanCamp , Tom Holland , Frank Grillo , William Hurt , and Daniel Brühl . In Captain America : Civil
War , disagreement over international oversight of the Avengers fractures them into opposing factions -- one led by Steve Rogers and the other by Tony Stark .
Development of Civil War began in late 2013 when Markus and McFeely began writing the screenplay , which borrows concepts from the 2006 comic book storyline ``
Civil War '' , while also focusing on story and character elements from the previous Captain America films to conclude the trilogy .
</item 1>

<item 2>
Captain America: Civil War
^ Jump up to : Brett , Jennifer ( May 11 , 2015 ) . `` Photos from the `` Captain America : Civil War '' set in Atlanta `` . Atlanta Journal - Constitution .
Archived from the original on May 12 , 2015 . Retrieved May 12 , 2015 . Jump up ^ Brett , Jennifer ( May 13 , 2015 ) . `` Photos from the `` Captain America :
Civil War '' set in Atlanta `` . Atlanta Journal - Constitution . Archived from the original on May 14 , 2015 . Retrieved May 14 , 2015 . Jump up ^ Brett ,
Jennifer ( May 15 , 2015 ) . `` Photos : Chris Evans on the Atlanta `` Captain America '' set `` . Atlanta Journal - Constitution . Archived from the original
on May 17 , 2015 . Retrieved May 15 , 2015 . ^ Jump up to : Leslie , Jennifer ( May 5 , 2016 ) . `` '' Captain America : Civil War '' was shot in , around
Atlanta `` . WXIA - TV .
</item 2>

<item 3>
Captain America: Civil War
Captain America : Civil War - Wikipedia Captain America : Civil War Jump to : navigation , search Captain America : Civil War Theatrical release poster Directed
by Anthony Russo Joe Russo Produced by Kevin Feige Screenplay by Christopher Markus Stephen McFeely Based on Captain America by Joe Simon Jack Kirby Starring
Chris Evans Robert Downey Jr . Scarlett Johansson Sebastian Stan Anthony Mackie Don Cheadle Jeremy Renner Chadwick Boseman Paul Bettany Elizabeth Olsen Paul
Rudd Emily VanCamp Tom Holland Frank Grillo William Hurt Daniel Brühl Music by Henry Jackman Cinematography Trent Opaloch Edited by Jeffrey Ford Matthew Schmidt
Production company Marvel Studios Distributed by Walt Disney Studios Motion Pictures Release date April 12 , 2016 ( 2016 - 04 - 12 ) ( Dolby Theatre ) May 6 ,
2016 ( 2016 - 05 - 06 ) ( United States ) Running time 147 minutes Country United States Language English Budget $250 million Box office $1.153 billion Captain
America : Civil War is a 2016 American superhero film based on the Marvel Comics character Captain America , produced by Marvel Studios and distributed by Walt
Disney Studios Motion Pictures .
</item 3>

<item 4>
Captain America: Civil War
Contents ( hide ) 1 Plot 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 3.4. 1 Visual effects 4 Music 5 Release 5.1
Marketing 5.1. 1 Merchandise 5.2 Home media 6 Reception 6.1 Box office 6.1. 1 United States and Canada 6.1. 2 Other territories 6.2 Critical response 6.3
Accolades 7 Future 8 Notes 9 References 10 External links Plot ( edit ) In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a
Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in
the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a
biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers .
</item 4>

<item 5>
Captain America: Civil War
Following positive reactions to test screenings of The Winter Soldier , the Russo brothers were brought back to direct in early 2014 . The film 's title and
premise were revealed in October 2014 , along with Downey 's casting ; additional cast members joined in the following months . Principal photography began in
April 2015 at Pinewood Atlanta Studios in Fayette County , Georgia , and continued in the Metro Atlanta area before concluding in Germany in August 2015 , with
the film being the first to use IMAX 's digital 2D cameras ( for the film 's central airport fight sequence ) . Visual effects were provided by nearly 20
different studios during the post-production process . Captain America : Civil War held its world premiere in Los Angeles on April 12 , 2016 , and was released
in the United States on May 6 , 2016 , in 3D and IMAX 3D . The film became a critical and commercial success , grossing over $1.1 billion worldwide , making it
the highest - grossing film of 2016 and , at the time , the 12th - highest - grossing film ever .
</item 5>

<item 6>
Captain America: Civil War
Filming locations in Germany included the Olympic Stadium in Berlin , and the Leipzig / Halle Airport in Schkeuditz . Additional filming also took place in
Puerto Rico and Norway , and was scheduled to in Iceland . Principal photography wrapped on August 22 , 2015 . Captain America : Civil War was the first film to
use IMAX 's digital 2D cameras , made in partnership with Arri , which was a customized version of the Arri Alexa 65 . According to Joe Russo , approximately
fifteen minutes of the film , the sequence where the two factions fight at Leipzig / Halle Airport , which was described as having `` incredible scale to it ''
and referred to by the directors as the `` splash panel '' , were shot with the cameras . Opaloch noted that there were discussions late in pre-production to
shoot the entire film with the Alexa 65 cameras . However , because of how close it was to the start of shooting as well as the idea being `` such a large -
scale undertaking '' , it was decided to only use them for the airport sequence , with the rest of film being shot on Arri Alexa XT digital cameras .
</item 6>

<item 7>
Captain America: Civil War
wikipedia.org/w/index.php?title=Captain_America:_Civil_War&oldid=819487324 '' Categories : 2016 films English - language films 2010s action thriller films 2010s
adventure films 2010s psychological thriller films 2010s science fiction films 2010s superhero films 2016 3D films American 3D films American action thriller
films American films American political thriller films American psychological thriller films American science fiction action films American science fiction
thriller films American sequel films Babelsberg Studio films Captain America films Crossover films Drone films Films scored by Henry Jackman Films about revenge
Films about terrorism Films based on works by Mark Millar Films directed by Anthony and Joe Russo Films set in 1991 Films set in Berlin Films set in Bucharest
Films set in Cleveland Films set in Germany Films set in Lagos Films set in London Films set in Massachusetts Films set in New York ( state ) Films set in
Queens , New York Films set in Siberia Films set in Vienna Films set in prison Films set in a fictional African country Films shot at Pinewood Atlanta Studios
Films shot in Atlanta Films shot in Berlin Films shot in Brazil Films shot in Germany Films shot in Norway Films shot
</item 7>

<item 8>
Captain America: Civil War
The film 's Southeast Asia premiere was held on April 21 at the Marina Bay Sands resort in Singapore , while the European premiere took place at Vue Cinemas in
Westfield London on April 26 . The film was released internationally starting from April 27 , releasing in 61 countries its first weekend , including the United
Kingdom on April 29 . The North America release on May 6 , took place in over 4,200 theaters , of which 3,300 were in 3D , along with 378 IMAX theaters , 480
premium large - format , and 161 D - Box locations . Internationally , the film opened in 955 IMAX theaters , while South Korea saw Civil War open in an ``
unprecedented '' 1,989 theaters . In September 2014 , TNT acquired the cable broadcast rights for Captain America : Civil War to air two years after its
theatrical release . Marketing ( edit ) In early July 2015 , Marvel began a viral marketing campaign for Ant - Man featuring Leslie Bibb , reprising her role
from the Iron Man films as journalist Christine Everhart , reporting for the faux news program , WHIH Newsfront .
</item 8>

<item 9>
Captain America: Civil War
Retrieved May 26 , 2016 . Jump up ^ Romano , Nick ( August 22 , 2015 ) . `` See the Captain America : Civil War cast celebrate wrapping production '' .
Entertainment Weekly . Archived from the original on August 23 , 2015 . Retrieved August 23 , 2015 . Jump up ^ Perry , Spencer ( August 22 , 2015 ) . `` Captain
America : Civil War Wraps Principal Photography '' . Superhero Hype ! . Archived from the original on August 22 , 2015 . Retrieved August 22 , 2015 . Jump up ^
Giardina , Carolyn ( May 6 , 2016 ) . `` How the ' Captain America : Civil War ' Airport Battle Was Filmed and Why It 's a Prelude to ' Avengers : Infinity War
' '' . The Hollywood Reporter . Archived from the original on May 16 , 2016 . Retrieved May 15 , 2016 . Jump up ^ Perry , Spencer ( September 3 , 2015 ) .
</item 9>

<item 10>
Captain America: Civil War
The next month , it was revealed that the film would be converted to 3D in post-production , and that Brühl would be playing Helmut Zemo . Additionally ,
Elizabeth Olsen revealed she would reprise her role in the film as Wanda Maximoff / Scarlet Witch . Filming ( edit ) Film set for Captain America : Civil War in
Downtown Atlanta Principal photography began on April 27 , 2015 , in Fayette County , Georgia at Pinewood Atlanta Studios , under the working title Sputnik .
Other filming locations in the Atlanta metropolitan area included the Buckhead district of Atlanta ; the Peachtree Christian Church in Midtown Atlanta ;
Downtown Atlanta , including an area near Philips Arena known as The Gulch , which served as the market in Lagos ; Norcross , Georgia ; Porsche 's headquarters
in Atlanta at Aerotropolis Atlanta , which served as the Avengers ' headquarters in the film ; and the Atlanta Civic Center , which served as the IFID (
Institute for Infectious Diseases ) laboratory in Lagos ; the latter 's interiors doubled as MIT , and a third part was used for a Berlin location .
</item 10>
=== END ITEM id=8890487e8d2009d2 ===
