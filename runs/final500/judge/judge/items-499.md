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

This file has 12 items.

=== ITEM id=1f32b713725b5195 ===
Gold sentences and facts:
Sentence 0: Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister, and for two terms President of the General Assembly of the League of Nations (1930–32).
  Fact 0.0: Nicolae Titulescu was born on March 4, 1882.
  Fact 0.1: Nicolae Titulescu died on March 17, 1941.
  Fact 0.2: Nicolae Titulescu was a well-known Romanian diplomat.
  Fact 0.3: Nicolae Titulescu was at various times a government minister.
  Fact 0.4: Nicolae Titulescu was at various times finance minister.
  Fact 0.5: Nicolae Titulescu was at various times foreign minister.
  Fact 0.6: Nicolae Titulescu served for two terms as President of the General Assembly of the League of Nations.
  Fact 0.7: Nicolae Titulescu's two terms as President of the General Assembly of the League of Nations are dated 1930–32.
Sentence 1: The League of Nations (abbreviated as LN in English, "La Société des Nations " ] abbreviated as SDN or SdN in French) was an intergovernmental organisation founded on 10 January 1920 as a result of the Paris Peace Conference that ended the First World War.
  Fact 1.0: The League of Nations is abbreviated as LN in English.
  Fact 1.1: The League of Nations is abbreviated as SDN or SdN in French.
  Fact 1.2: The French name of the League of Nations is 'La Société des Nations'.
  Fact 1.3: The League of Nations was an intergovernmental organisation.
  Fact 1.4: The League of Nations was founded on 10 January 1920.
  Fact 1.5: The League of Nations was founded as a result of the Paris Peace Conference.
  Fact 1.6: The Paris Peace Conference ended the First World War.

Retrieved items:
<item 1>
Remus Nicolai
Remus Nicolai (b. June 10, 1977 in Bistriţa, Romania) is a retired Romanian aerobic gymnast.

 He had a successful career winning six world championships medals (one gold, two silver and three bronze) and five European championships medals (two gold, two
silver and one bronze).

 After retiring from aerobic gymnastics he opened together with his wife (Daniela Mărănducă) a private gymnastics club in Constanţa.

 As of 2013 he trains, together with his wife, the junior artistic women's National Team of Romania at the National Olympic Center in Onesti.
</item 1>

<item 2>
Gheorghe Tătărescu
During his first time in office, he moved closer to King Carol II, leading an ambivalent policy toward the fascist Iron Guard and ultimately becoming
instrumental in establishing the authoritarian and corporatist regime around the National Renaissance Front.

 In 1940, he accepted the cession of Bessarabia and Northern Bukovina to the Soviet Union, and consequently had to resign.
</item 2>

<item 3>
Ioan Silviu Suciu
Ioan Silviu Suciu (born 24 November 1977 in Sibiu) is a retired Romanian artistic gymnast.

 His best event was the pommel horse.

 One of his closest rivals was his team mate and pommel horse Olympic champion Marius Urzică, who was defeated by Suciu in Ljubljana, 2004, when the latter won
the title while Urzică took 8th place.

 Suciu is an Olympic bronze medalist with the team, a silver world medalist on pommel horse and a six-time European medalist (pommel horse, vault and team).

 Suciu was one of the key team members of the Romanian gymnastics team for several years contributing to the 2004 Olympic team bronze medal and three
continental team medals (two gold and one silver).

 He also placed fourth all around at the 2004 Olympic Games.
</item 3>

<item 4>
Ghimpețeni
Ghimpețeni is a commune in Olt County, Romania.

 It is composed of two villages, Ghimpețeni and Ghimpețenii Noi.

 These were part of Nicolae Titulescu Commune until 2004, when they were split off.
</item 4>

<item 5>
Theatre Organization of Cyprus (THOC), founded in 1970, promotes theatre and international artistic relations; it is governed by a presidentially appointed
board. Copenhagen’s Tivoli Concert Hall, designed by Frits Schlegel and Hans Hansen, opened in 1956. Czech dissident, playwright and writer Václav Havel became
Czechoslovakia’s last president (1989–92) and the Czech Republic’s first (1993–2003). A
</item 5>

<item 6>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 6>

<item 7>
Gheorghe Tătărescu
Gheorghe I. Tătărescu (also known as "Guță Tătărescu", with a slightly antiquated pet form of his given name; 2 November 1886 – 28 March 1957) was a Romanian
politician who served 36th Prime Minister of Romania (1934–1937; 1939–1940), three times as Minister of Foreign Affairs ("interim" in 1934 and 1938; appointed
to the office in 1945-1947), and once as Minister of War (1934).

 Representing the "young liberals" faction inside the National Liberal Party (PNL), Tătărescu began his political career as a collaborator of Ion G. Duca,
becoming noted for his anti-Communism and, in time, for his conflicts with the PNL's leader Dinu Brătianu and the Foreign Minister Nicolae Titulescu.
</item 7>

<item 8>
Profiles span journalists, artists, politicians, scholars and public figures: Nikola Mladenov founded Macedonia’s respected Fokus and inspired an
investigative-journalism award; Johannes Grenzfurthner leads art group monochrom; François Genoud financed Nazi networks and Middle Eastern militants; Edmund de
Unger collected Islamic art; Adel Ferdosipour hosts Iran’s Navad; Hans Aschenborn painted African wildlife; Katarina Ivanovska models and acts. Nicolae
Titulescu
</item 8>

<item 9>
Nicolae Titulescu
Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister,
and for two terms President of the General Assembly of the League of Nations (1930–32).
</item 9>

<item 10>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
</item 10>
=== END ITEM id=1f32b713725b5195 ===

=== ITEM id=d1e3adb7c940f193 ===
Gold sentences and facts:
Sentence 0: The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic Beach.
  Fact 0.0: The Nassau Herald is a weekly newspaper.
  Fact 0.1: The Nassau Herald serves the Five Towns communities of Nassau County.
  Fact 0.2: The Five Towns communities of Nassau County are Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood, and Atlantic Beach.
Sentence 1: Lawrence is a village in Nassau County, New York in the United States.
  Fact 1.0: Lawrence is a village.
  Fact 1.1: Lawrence is in Nassau County, New York, United States.
Sentence 2:  As of the 2010 United States Census, the village population was 6,483.
  Fact 2.0: As of the 2010 United States Census, the population of the village of Lawrence was 6,483.

Retrieved items:
<item 1>
Great Neck, NY, encompasses nine North Hempstead villages/hamlets (~40,000 residents), five ZIP codes and one school district; Saddle Rock is its Blue Ribbon
K–5 school. Floral Park village had 15,863 residents in 2010. Old Country Road serves central Nassau shopping and partly borders Hempstead/North Hempstead. Long
Island’s 75-mile outer barrier separates lagoons from the Atlantic. The Nassau Herald serves Five Towns. Other entries describe
</item 1>

<item 2>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 2>

<item 3>
Floral Park, New York
Floral Park is an incorporated village in Nassau County, New York, United States, on Long Island.

 The neighborhood of Floral Park in the New York City borough of Queens, is adjacent to the village.

 The village is at the western border of Nassau County, and is located mainly in the Town of Hempstead, while the section north of Jericho Turnpike is within
the Town of North Hempstead.

 The population as of the US Census of 2010 is 15,863.
</item 3>

<item 4>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 4>

<item 5>
Anton Media Group
Anton Media Group, formerly Anton Community Newspapers, and also known by its legal name Long Island Community Newspapers, Inc. is a print media company based
in Mineola, NY on Long Island in Nassau County and produces 17 weekly newspapers in Nassau County as well as "Long Island Weekly" and special sections such as
Healthy Living, Camp & School, and Dining Guides.

 Anton Community Newspapers is one of the largest privately owned newspaper companies in New York State.
</item 5>

<item 6>
Lawrence, Nassau County, New York
Lawrence is a village in Nassau County, New York in the United States.

 As of the 2010 United States Census, the village population was 6,483.
</item 6>

<item 7>
The Hudson Reporter
The Hudson Reporter is a newspaper chain based in Hudson County, New Jersey.

 It is the only weekly newspaper chain in Hudson County and one of only two newspaper companies in this busy metropolitan area.

 "The Hudson Reporter" publications focus on local politics and community news.

 In addition to articles written by the staff, the papers print readers' letters to the editor.

 The oldest newspaper in the chain is the "Hoboken Reporter", founded in 1983.
</item 7>

<item 8>
Nassau Herald
The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic
Beach.

 It is part of the Long Island Herald newspaper chain, which includes The Jewish Star and The Oyster Bay Guardian is owned by Richner Communications, and covers
Nassau County, New York.
</item 8>

<item 9>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 9>

<item 10>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 10>
=== END ITEM id=d1e3adb7c940f193 ===

=== ITEM id=472514dc316856e4 ===
Gold sentences and facts:
Sentence 0: Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.
  Fact 0.0: Brett Dallas McClure was born on February 19, 1981.
  Fact 0.1: Brett Dallas McClure was born in Yakima, Washington.
  Fact 0.2: Brett Dallas McClure is a retired American gymnast.
Sentence 1:  He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the team competition at the 2004 Summer Olympics in Athens.
  Fact 1.0: Brett Dallas McClure won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia.
  Fact 1.1: Brett Dallas McClure later helped his U.S. gymnastics team earn a silver medal in the team competition at the 2004 Summer Olympics in Athens.
Sentence 2:  The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.
  Fact 2.0: The competition was held in Brisbane, Queensland, Australia.
  Fact 2.1: The competition was held from 29 August to 9 September 2001.

Retrieved items:
<item 1>
Tabcorp Park Menangle
Tabcorp Park Menangle is a harness racing track in Sydney.

 Their main races are the Miracle Mile Pace, the NSW Derby, the NSW Oaks, the Len Smith Mile and the qualifiers for the Miracle Mile, the Allied Express Sprint
and the Canadian Club Sprint.

 It replaced Harold Park as Sydney's premier harness track in 2010.

 Fred Hastings and Anthony Manton are two callers that have called at the track before.
</item 1>

<item 2>
Klete Keller
Klete D. Keller (born March 21, 1982) is an American former competition swimmer who won medals at the 2000 Summer Olympics and the 2004 Summer Olympics in the
400-meter freestyle and the 4×200-meter freestyle relay.

 In the 4×200-meter freestyle relay, Keller held off a charging Ian Thorpe in the anchor leg to win the race by 0.13 seconds.

 This was the first time Australia had been beaten in the event in over seven years.

 The American relay of Michael Phelps, Ryan Lochte, Peter Vanderkaay, and Keller are undefeated since the Athens games.

 Vanderkaay, Larsen Jensen, Erik Vendt, and Keller make up the core of the premier American mid-distance/distance freestyle swimmers.
</item 2>

<item 3>
Anthony Biddle (Paralympian)
Anthony John Biddle, OAM (born 18 June 1975) is an Australian Paralympic tandem cyclist and athlete.

 He was born in the New South Wales city of Gosford.

 He competed in athletics without winning any medals at the 1996 Atlanta Games and the 2000 Sydney Games.

 At the 2004 Athens Games, he switched to cycling and won a gold medal in the Men's 1 km Time Trial Tandem B1–3 event, for which he received a Medal of the
Order of Australia, and a bronze medal in the Men's Sprint Tandem B1–3 event.

 Kial Stewart was his pilot for both events.
</item 3>

<item 4>
2001 Goodwill Games
The 2001 Goodwill Games was the fifth and final edition of the international multi-sport event.

 The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.

 A total around 1300 athletes took part in 14 sporting competitions.
</item 4>

<item 5>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 5>

<item 6>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 6>

<item 7>
2002 Commonwealth Games
The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25
July to 4 August 2002.

 The 2002 Games were to be hosted in the United Kingdom to coincide with the Golden Jubilee of Elizabeth II, head of the Commonwealth, and Manchester was
selected for the 2002 Games ahead of London.

 The XVII Commonwealth Games was, prior to the 2012 Summer Olympics, the largest multi-sport event ever to be held in the UK, eclipsing the London 1948 Summer
Olympics in numbers of teams and athletes participating.

 In terms of sports and events, the 2002 Games were the largest Commonwealth Games in history featuring 281 events across 17 sports.
</item 7>

<item 8>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 8>

<item 9>
Brett McClure
Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.

 He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the
team competition at the 2004 Summer Olympics in Athens.

 During his sporting career, McClure has collected two more silver medals in the same program at the World Artistic Gymnastics Championships (2001 and 2003).

 McClure is currently the High Performance Director with USA men's gymnastics.
</item 9>

<item 10>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 10>
=== END ITEM id=472514dc316856e4 ===

=== ITEM id=1519a313a7d5f226 ===
Gold sentences and facts:
Sentence 0: The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of its kind in Northern Europe.
  Fact 0.0: The Cathedral and Metropolitical Church of Saint Peter in York is commonly known as York Minster.
  Fact 0.1: York Minster is the cathedral of York, England.
  Fact 0.2: York Minster is one of the largest cathedrals of its kind in Northern Europe.
Sentence 1: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.
  Fact 1.0: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

Retrieved items:
<item 1>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 1>

<item 2>
Minster (church)
Minster is an honorific title given to particular churches in England, most famously York Minster in York, Westminster in London and Southwell Minster in
Southwell.

 The term "minster" is first found in royal foundation charters of the 7th century.

 Although it corresponds to the Latin "monasterium" or monastery, it then designated any settlement of clergy living a communal life and endowed by charter with
the obligation of maintaining the daily office of prayer.

 Widespread in 10th-century Anglo-Saxon England, minsters declined in importance with the systematic introduction of parishes and parish churches from the 11th
century onwards.

 It continued as a title of dignity in later medieval England, for instances where a cathedral, monastery, collegiate church or parish church had originated
with an Anglo-Saxon foundation.

 Eventually a minster came to refer more generally to "any large or important church, especially a collegiate or cathedral church".

 In the 21st century, the Church of England has designated additional minsters by bestowing the status on existing parish churches.
</item 2>

<item 3>
Old Palace (York)
The Old Palace in the city of York, North Yorkshire, England, is also known as the Minster Library and is in Dean's Park.

 It houses York Minster’s library and archives as well as the Collections Department and conservation studio.

 Its name is a new one and renders homage to the part of the building that used to be the chapel of the Archbishop of York, which was built in the 13th century.

 It was refurbished in 1810 and shortly thereafter became the home of the Minster library.

 Notable items held in the collection include cathedral records dating to back to 1150 and a copy of the 1631 Wicked Bible.

 It is a Grade I listed building.

 An extension was added in 1998.
</item 3>

<item 4>
York Minster Police
York Minster Police is a small, specialised cathedral constabulary responsible for policing York Minster in York, United Kingdom.
</item 4>

<item 5>
York
York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.

 The municipality is the traditional county town of the historic county of Yorkshire to which it gives its name.

 The city has a rich heritage and has provided the backdrop to major political events in England throughout much of its two millennia of existence.

 The city offers a wealth of historic attractions, of which York Minster is the most prominent, and a variety of cultural and sporting activities making it a
popular tourist destination for millions.
</item 5>

<item 6>
York Minster
The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of
its kind in Northern Europe.

 The minster is the seat of the Archbishop of York, the second-highest office of the Church of England, and is the mother church for the Diocese of York and the
Province of York.

 It is run by a dean and chapter, under the Dean of York.

 The title "minster" is attributed to churches established in the Anglo-Saxon period as missionary teaching churches, and serves now as an honorific title.

 Services in the minster are sometimes regarded as on the High Church or Anglo-Catholic end of the Anglican continuum.
</item 6>

<item 7>
York Minster astronomical clock
The York Minster astronomical clock was installed in the North Transept of York Minster in 1955.

 It was first conceived in 1944 and designed by R d'E Atkinson, chief assistant at the Royal Greenwich Observatory.

 The clock is a memorial to the airmen operating from bases in Yorkshire, Durham, and Northumberland who were killed in action during World War II.
</item 7>

<item 8>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 8>

<item 9>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 9>

<item 10>
Grays Court, York
Grays Court is a Grade I listed ancient house in the middle of York, England, near York Minster, behind the Treasurer's House next to the city walls.

 Grays Court is possibly the oldest continuously occupied house in the United Kingdom.

 Dating back in part to 1080 and commissioned by the first Norman Archbishop of York to provide the official residence for the Treasurers of York Minster, the
house has a significant history.

 The house was surrendered to the Crown on 26 May 1547 and the last of the mediaeval Treasurers, William Clyff, resigned.

 The first post-Reformation owner was Edward Seymour, Duke of Somerset.

 He was given the house in 1547 by King Edward VI, the son of King Henry VIII.
</item 10>
=== END ITEM id=1519a313a7d5f226 ===

=== ITEM id=1977f7af3c0a68c0 ===
Gold sentences and facts:
Sentence 0: Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit single in 1986 with "Love Missile F1-11" – and six other EMI single releases.
  Fact 0.0: Martin Degville was born 27 January 1961 in Walsall.
  Fact 0.1: Martin Degville is the lead singer of the UK pop band Sigue Sigue Sputnik.
  Fact 0.2: Martin Degville is a co-songwriter of Sigue Sigue Sputnik.
  Fact 0.3: Sigue Sigue Sputnik had a worldwide hit single in 1986 with 'Love Missile F1-11'.
  Fact 0.4: Sigue Sigue Sputnik has six other EMI single releases.
Sentence 1: Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.
  Fact 1.0: Paul Richard 'Richie' Furay was born May 9, 1944.
  Fact 1.1: Richie Furay is an American singer.
  Fact 1.2: Richie Furay is an American songwriter.
  Fact 1.3: Richie Furay is a Rock & Roll Hall of Fame member.
  Fact 1.4: Richie Furay is best known for forming Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin.
  Fact 1.5: Richie Furay is best known for forming Poco with Jim Messina, Rusty Young, George Grantham, and Randy Meisner.

Retrieved items:
<item 1>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
</item 1>

<item 2>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 2>

<item 3>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 3>

<item 4>
The material covers country, country-rock, folk-rock and bluegrass artists and releases. Oak Ridge Boys’ 2009 *The Boys Are Back* marked a return to secular
country, led by a White Stripes cover. Richie Furay helped form Buffalo Springfield and Poco, later joining the Souther–Hillman–Furay Band; Poco’s debut
featured his “Pickin’ Up the Pieces.” Other subjects include Buffalo Springfield’s final album, the Lonesome Pine F
</item 4>

<item 5>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 5>

<item 6>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 6>

<item 7>
Dean Martin Sings
Dean Martin Sings is the first studio album by Italian-American singer Dean Martin, released in 1953.

 It is the first long-play 10-inch album recorded by Martin for Capitol Records during two sessions recorded on the evening of November 20, 1952.

 The first session was recorded between 5 and 8 PM and it produced five songs featuring string arrangements.

 "There's My Lover" was recorded but not released.

 After a ninety-minute break, Martin was joined by a brass arrangement to record the remaining four songs.

 Seven of the eight songs on this album appeared in the Martin & Lewis film, "The Stooge".

 Two years later, the songs from this 10-inch album would be combined with four newly popular songs recorded between 1951 and 1953 to create a full-length
12-inch album.

 The 2005 Collectors' Choice reissue added four bonus songs recorded between 1949 and 1953 and was released with alternative cover artwork.
</item 7>

<item 8>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 8>

<item 9>
Martin Degville
Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit
single in 1986 with "Love Missile F1-11" – and six other EMI single releases.

 Sputnik was formed with ex-Generation X bassist Tony James.
</item 9>

<item 10>
Poco: The Songs of Richie Furay
Poco: The Songs of Richie Furay is a compilation album consisting of songs by Richie Furay during his tenure with the band Poco, released in 1980.
</item 10>
=== END ITEM id=1977f7af3c0a68c0 ===

=== ITEM id=b2f53c3143d9fb90 ===
Gold sentences and facts:
Sentence 0: Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
  Fact 0.0: Gunnar Eric Nelson was born on September 20, 1967.
  Fact 0.1: Gunnar Eric Nelson is an American musician.
  Fact 0.2: Gunnar Eric Nelson is an American singer.
  Fact 0.3: Gunnar Eric Nelson is an American songwriter.
  Fact 0.4: Gunnar Eric Nelson is an international multi-platinum recording artist.
Sentence 1: Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the Machine.
  Fact 1.0: Florence Leontine Mary Welch was born on 28 August 1986.
  Fact 1.1: Florence Leontine Mary Welch is an English musician.
  Fact 1.2: Florence Leontine Mary Welch is an English singer.
  Fact 1.3: Florence Leontine Mary Welch is an English songwriter.
  Fact 1.4: Florence Leontine Mary Welch is best known as the vocalist of the indie rock band Florence and the Machine.

Retrieved items:
<item 1>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
</item 1>

<item 2>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 2>

<item 3>
Gunnar Nelson (musician)
Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
</item 3>

<item 4>
Bo Welch
Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
</item 4>

<item 5>
Sweet Nothing
"Sweet Nothing" is a song by Scottish recording artist and record producer Calvin Harris from his third studio album, "18 Months" (2012).

 It features vocals from Florence and the Machine singer Florence Welch.

 Harris previously worked with the band in a remix of their single, "Spectrum" (2012).

 While recording "18 Months", Harris expressed interest in working with Welch.

 He mentioned that it was not easy and he had to chase her, due to schedule conflicts.

 Welch accepted his invitation and the two recorded "Sweet Nothing".

 The track premiered on "The Chris Moyles Show" on BBC Radio 1 on 28 August 2012 and was later released as the fifth single from the album on 12 October 2012.
</item 5>

<item 6>
Nelson (band)
Nelson is an American rock band founded by singer/songwriters Matthew and Gunnar Nelson (twin sons of Ricky Nelson and Kristin Nelson).

 The band achieved success during the early 1990s with their double platinum debut album "After the Rain", which featured the number-one hit "(Can't Live
Without Your) Love and Affection".
</item 6>

<item 7>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 7>

<item 8>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 8>

<item 9>
Romeo Nelson
Iromeio "Romeo" Nelson (March 12, 1902 – May 17, 1974) was an American boogie woogie pianist whose recordings from 1929 are regarded as some of the finest, and
certainly the fastest, boogie woogie showpieces on record.
</item 9>

<item 10>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 10>
=== END ITEM id=b2f53c3143d9fb90 ===

=== ITEM id=89b2fd516e7eca48 ===
Gold sentences and facts:
Sentence 0: Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
  Fact 0.0: Rolf Martin Theodor Dahlgren was born on 7 July 1932.
  Fact 0.1: Rolf Martin Theodor Dahlgren died on 14 February 1987.
  Fact 0.2: Rolf Martin Theodor Dahlgren was a Swedish-Danish botanist.
  Fact 0.3: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen.
  Fact 0.4: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen from 1973 until his death.
Sentence 1:  The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.
  Fact 1.0: The university has 23,473 undergraduate students.
  Fact 1.1: The university has 17,398 postgraduate students.
  Fact 1.2: The university has 2,968 doctoral students.
  Fact 1.3: The university has over 9,000 employees.

Retrieved items:
<item 1>
Dimitrije Đorđević (historian)
Đorđević was awarded his doctorate in 1962.

 In 1970, Đorđević took up a position as a Full Professor of History at the University of California, Santa Barbara, joining a strong faculty in European
History including Joachim Remak, Leonard Marsak, and C. Warren Hollister.

 He was elected a member of the Serbian Academy of Science and Arts in 1985.

 A popular undergraduate lecturer and graduate mentor at the University of California, Santa Barbara, in 1992 many of his former students contributed to his
"Festschrift" entitled "Scholar, Patriot, Mentor: Historical Essays in Honor of Dimitrije Djordjevic".

 In retirement, Đorđević published his autobiography, "Scars and Memory: Four Lives in One Lifetime", describing his World War II and post World War II
experiences.

 Professor Đorđević died in Santa Barbara on March 5, 2009.
</item 1>

<item 2>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 2>

<item 3>
Walt Disney's Treasury of Classic Tales
Walt Disney's Treasury of Classic Tales is an American Sunday comic strip, which ran in newspapers from 13 July 1952 until 15 February 1987.

 Each story adapted a different Disney film, such as Darby O'Gill and the Little People, Peter Pan, or Davy Crockett.

 It was run in relatively few papers with 58 in 1957 and 55 in 1966 and was principally a vehicle for promoting new and re-released Disney films.
</item 3>

<item 4>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 4>

<item 5>
Richard W. Aldrich
Richard Warren Aldrich is an American neuroscientist who is currently the Karl Folkers Chair of Interdisciplinary Medical Research and Professor of Neurobiology
in the School of Biological Sciences at the University of Texas.

 Graduating from the University of Arizona with a BS in Biological Sciences (1975), Aldrich went on to earn a Ph. D. in Neuroscience from Stanford University
(1980).

 After completing a post-doctoral fellowship in physiology at Yale University under the direction of W. Knox Chandler and Charles F. Stevens, Aldrich was hired
to teach at Yale, instructing in the Department of Molecular Neurobiology.

 In 1985, Aldrich returned to Stanford to teach neurobiology and physiology, eventually serving as Chair of the Department of Molecular and Cellular Physiology
from 2001 to 2004.

 He was an Investigator in the Howard Hughes Medical Institute from 1990 until 2006.

 Aldrich's research has focused on the "molecular mechanisms of ion channel function and their role in electrical signaling."

 Aldrich has served as the President of the Society of General Physiologists and the Biophysical Society.
</item 5>

<item 6>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 6>

<item 7>
Rolf Dahlgren
Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
</item 7>

<item 8>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 8>

<item 9>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 9>

<item 10>
Gertrud Dahlgren
Gertrud Dahlgren (1931–2009) was a Swedish botanist.

 She was born at Klippan in Scania, and after graduating from university at Helsingborg went to the University of Lund for graduate studies.

 There she obtained her M.Sc.

 in chemistry and biology.

 She was married to fellow botanist Rolf Dahlgren (1932–1987), who was killed in a car accident.

 The couple had three children, Susanna, Helena and Fredrik.

 She died in December 2009 at the age of 78.
</item 10>
=== END ITEM id=89b2fd516e7eca48 ===

=== ITEM id=9bb7004942047317 ===
Gold sentences and facts:
Sentence 0: Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.
  Fact 0.0: Marcel Émile Verdet was born on 13 March 1824.
  Fact 0.1: Marcel Émile Verdet died on 3 June 1866.
  Fact 0.2: Marcel Émile Verdet was a French physicist.
Sentence 1:  Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de physique".
  Fact 1.0: Marcel Émile Verdet did much to champion the early theory of the conservation of energy in France.
  Fact 1.1: Marcel Émile Verdet championed this theory in France through his editorial supervision of the 'Annales de chimie et de physique'.
Sentence 2: Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the title "Annales de chimie".
  Fact 2.0: Annales de chimie et de physique is French for 'Annals of Chemistry and of Physics'.
  Fact 2.1: Annales de chimie et de physique is a scientific journal.
  Fact 2.2: Annales de chimie et de physique was founded in Paris, France, in 1789.
  Fact 2.3: Annales de chimie et de physique was founded under the title 'Annales de chimie'.

Retrieved items:
<item 1>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 1>

<item 2>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 2>

<item 3>
Frances Liardet
Frances E. Liardet is a writer and translator of Arabic literature.

 She studied Creative Writing at the University of East Anglia.

 She has translated several book-length works, including two books by the modernist Egyptian writer Edwar al-Kharrat and one by Nobel Prize winner Naguib
Mahfouz.
</item 3>

<item 4>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 4>

<item 5>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 5>

<item 6>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 6>

<item 7>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 7>

<item 8>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 8>

<item 9>
Villard (imprint)
Villard, also known as Villard Books, is a publishing imprint of Random House, one of the largest publishing companies in the world.

 It was founded in 1983.

 Villard began as an independent imprint of Random House and is currently a sub-imprint of Ballantine Books, itself an imprint of Random House.

 It was named after a Stanford White brownstone mansion on Madison Avenue that was the home of Random House for twenty years.
</item 9>

<item 10>
Index of Middle English Verse
The Index of Middle English Verse (IMEV) is a bibliographic index of poetry in Middle English.

 Its first print publication, in 1943, was an extension of Carleton Brown's "Register of Middle English Religious & Didactic Verse", augmented by the inclusion
of secular verse.

 This edition, edited by Brown and Rossell Hope Robbins, contained entries for over 4000 Middle English poems in more than 2000 manuscripts.

 In 1965 the index was supplemented by Robbins and John L. Cutler.

 The IMEV is available online through Virginia Polytechnic Institute and State University.
</item 10>
=== END ITEM id=9bb7004942047317 ===

=== ITEM id=11d331ef1821ea5c ===
Gold sentences and facts:
Sentence 0: Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and Randall Counties, Texas, United States.
  Fact 0.0: Rick Husband Amarillo International Airport has the IATA code AMA.
  Fact 0.1: Rick Husband Amarillo International Airport has the ICAO code KAMA.
  Fact 0.2: Rick Husband Amarillo International Airport has the FAA LID AMA.
  Fact 0.3: Rick Husband Amarillo International Airport is a public airport.
  Fact 0.4: Rick Husband Amarillo International Airport is six miles (10 km) east of downtown Amarillo.
  Fact 0.5: Rick Husband Amarillo International Airport is in Potter and Randall Counties, Texas, United States.
Sentence 1: Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County, North Dakota.
  Fact 1.0: Grand Forks International Airport has the IATA code GFK.
  Fact 1.1: Grand Forks International Airport has the ICAO code KGFK.
  Fact 1.2: Grand Forks International Airport has the FAA LID GFK.
  Fact 1.3: Grand Forks International Airport is a public airport.
  Fact 1.4: Grand Forks International Airport is five miles (8 km) northwest of Grand Forks.
  Fact 1.5: Grand Forks International Airport is in Grand Forks County, North Dakota.

Retrieved items:
<item 1>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 1>

<item 2>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 2>

<item 3>
Grand Forks International Airport
Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County,
North Dakota.

 GFK has no scheduled passenger flights out of the country but has an "international" title (like many other airports) because it has customs service for
arrivals from Canada and other countries.
</item 3>

<item 4>
Grand Forks International
The Grand Forks International (GFI) is an annual international invitational baseball tournament hosted at James Donaldson Park in Grand Forks, British Columbia.

 The GFI is the largest invitational baseball tournament in Canada and is a large part of the culture in Grand Forks.

 The tournament is also notable for being run entirely by volunteers.
</item 4>

<item 5>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 5>

<item 6>
Emerado, North Dakota
Emerado is a city in Grand Forks County, North Dakota, United States located near Grand Forks Air Force Base.

 It is part of the "Grand Forks, ND-MN Metropolitan Statistical Area" or "Greater Grand Forks".

 The population was 414 at the 2010 census.

 Grand Forks Air Force Base is located near Emerado.
</item 6>

<item 7>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 7>

<item 8>
Lea County Regional Airport
Lea County Regional Airport (IATA: HOB, ICAO: KHOB) (Lea County-Hobbs Airport) is four miles (6.4 km) west of Hobbs, in Lea County, New Mexico.

 The airport covers 898 acre and has three runways.

 It is an FAA certified commercial airport served by United Airlines' affiliate with daily regional flights.

 Lea County Regional Airport is the largest of the three airports owned and operated by Lea County Government.

 Lea County also owns and operated two general aviation airports in Lovington and Jal, New Mexico.
</item 8>

<item 9>
Downtown Grand Forks
Downtown Grand Forks is the original commercial center of Grand Forks, North Dakota.

 Located on the western bank of the Red River of the North, the downtown neighborhood is situated near the fork of the Red River and the Red Lake River.

 While downtown is no longer the dominant commercial area of the Greater Grand Forks community, it remains the historic center of Grand Forks.

 An 80.4 acre portion was listed on the National Register of Historic Places in 2005, as Downtown Grand Forks Historic District.

 Today, downtown Grand Forks is home to many offices, stores, restaurants, and bars.
</item 9>

<item 10>
John D. Odegard School of Aerospace Sciences
The John D. Odegard School of Aerospace Sciences (UND Aerospace) is a multidisciplinary college within the University of North Dakota (UND) in Grand Forks,
North Dakota.

 The school was formed in 1968.

 The majority of the school's fleet of over 120 aircraft is based at nearby Grand Forks International Airport and is the largest fleet of civilian flight
training aircraft in North America.

 UND Aerospace also operates flight training centers in Crookston, Minnesota, and Phoenix, Arizona.

 Today, the school has many aerospace-related programs including commercial aviation (fixed wing and rotorcraft), air traffic control, airport management, Space
Studies, Computer Science, Atmospheric Sciences, and Earth System Science & Policy.

 Currently, the school has over 500 faculty and 1,900 students making it the second largest of UND’s degree-granting colleges.

 The present dean of the school is Dr. Paul Lindseth.
</item 10>
=== END ITEM id=11d331ef1821ea5c ===

=== ITEM id=2026a947b9f24640 ===
Gold sentences and facts:
Sentence 0:  Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college levels.
  Fact 0.0: The institution has three divisions.
  Fact 0.1: Through its three divisions, the institution offers a wide variety of liberal arts courses.
  Fact 0.2: Through its three divisions, the institution offers a wide variety of science courses.
  Fact 0.3: The courses are offered at the secondary level.
  Fact 0.4: The courses are offered at the British A-level level.
  Fact 0.5: The courses are offered at the United States junior college level.
Sentence 1:  Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in education.
  Fact 1.0: Grammar was a core liberal art.
  Fact 1.1: Logic was a core liberal art.
  Fact 1.2: Rhetoric was a core liberal art.
  Fact 1.3: Arithmetic also played a somewhat lesser part in education.
  Fact 1.4: Geometry also played a somewhat lesser part in education.
  Fact 1.5: The theory of music also played a somewhat lesser part in education.
  Fact 1.6: Astronomy also played a somewhat lesser part in education.

Retrieved items:
<item 1>
St. John's High School (South Carolina)
St. John's High School (SJHS) is a senior high school on Johns Island, South Carolina.

 It is a part of the Charleston County School District.

 St. John's is home to approximately 300 students and 30 faculty and staff.

 St. John's school mascot is The Mighty Islanders, sporting royal blue and maroon as the school colors.

 St. John's offers Advanced Placement and dual credit courses totaling at over 30 hours of offered college credit, as well as 3 career academies in Hospitality
and Tourism, Computer Science, and Culinary Arts.

 St. John's competes at the A level in football, volleyball, basketball (boys and girls), wrestling, soccer, track, baseball, and softball.

 The Islanders also offer marching band, agriculture and green house, weightlifting, competitive academic team, and student council.
</item 1>

<item 2>
Aalborghus Gymnasium
Aalborghus Gymnasium is an upper secondary school in the city of Aalborg, in North Jutland in Denmark.

 It offers both the traditional three-year program and also the two-year Higher Preparatory Examination (HF) program.

 The subjects taught at the school range from Religion and Music to Spanish and Natural Geography.

 Aalborghus Gymnasium attempts to focus on the musical and creative side of students as well as taking an international perspective on issues.

 Students begin their studies at Aalborghus by selecting a stream of studies.

 Each stream has two or three subjects that are the focus of the studies.

 English/Social Studies is one example of a stream a student can select.
</item 2>

<item 3>
Trinity University (Texas)
Trinity University is a private liberal arts college in San Antonio, Texas.

 Founded in 1869, its campus is located in the Monte Vista Historic District adjacent to Brackenridge Park.

 The campus is three miles north of downtown San Antonio and the River Walk and six miles south of the San Antonio International Airport.

 The student body consists of approximately 2,300 undergraduate and 200 graduate students.

 Trinity offers 42 majors and 57 minors among 6 degree programs and has an endowment of $1.1 billion, the 85th largest in the country, which permits it to
provide resources typically associated with much larger colleges and universities.
</item 3>

<item 4>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 4>

<item 5>
List of St. John's Seminary (California) people
The list of St. John's Seminary (California) people is a compilation of lists of notable alumni, faculty, and current students of St. John's Seminary in
Camarillo, California, United States.

 St. John's Seminary grants graduate degrees for seminarians preparing for the priesthood, as well as a graduate degree for lay persons interested in pastoral
ministry.

 The St. John's Seminary College was the undergraduate division of the seminary before it closed in the early 21st century.

 The table of notable alumni lists the date of graduation from St. John's college, seminary, or both, if applicable.

 It is not unusual for seminarians to have received their undergraduate education at a different institution than their seminary training.
</item 5>

<item 6>
St. John's College High School, Belize
St. John's College High School is a high school for boys situated in Belize City, Belize.

 It was founded in 1887.

 The High School exists to educate academically talented young men in a Jesuit environment of self-discipline, love of learning, and service to others.

 The school 's curriculum is complemented by sports and extracurricular activities.

 The third and fourth form classes follow the Caribbean Secondary Education Certificate (CSEC) curricula and sit the regionally administered examinations at the
end of their fourth year.
</item 6>

<item 7>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 7>

<item 8>
Liberal arts education
The liberal arts (Latin: "artes liberales") are those subjects or skills that in classical antiquity were considered essential for a free person (Latin:
"liberalis", "worthy of a free person") to know in order to take an active part in civic life, something that (for Ancient Greece) included participating in
public debate, defending oneself in court, serving on juries, and most importantly, military service.

 Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in
education.
</item 8>

<item 9>
St. John's College, Belize
St. John's College has three divisions, and a number of central academic centres and activities.

 Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college
levels.

 St. John's College is a Roman Catholic institution in the Jesuit tradition, one of the oldest, largest, and most diverse educational institutions in Belize,
founded by the Jesuits in 1887.
</item 9>

<item 10>
Lydia Patterson Institute
Lydia Patterson Institute is a Methodist Christian college-preparatory school located in El Paso, Texas, United States.

 Founded in 1913, it offers programs for Spanish-speaking children, primarily from Juarez, to attend high school in the United States and attend an American
undergraduate university.

 All high school classes are taught in English, and the school is accredited by the Southern Association of Colleges and Schools.
</item 10>
=== END ITEM id=2026a947b9f24640 ===

=== ITEM id=57f8c2b015344517 ===
Gold sentences and facts:
Sentence 0:  One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
  Fact 0.0: Updike was one of only three writers to win the Pulitzer Prize for Fiction more than once.
  Fact 0.1: The other two writers who won the Pulitzer Prize for Fiction more than once were Booth Tarkington and William Faulkner.
  Fact 0.2: Updike published more than twenty novels during his career.
  Fact 0.3: Updike published more than a dozen short-story collections during his career.
  Fact 0.4: Updike published poetry during his career.
  Fact 0.5: Updike published art and literary criticism during his career.
  Fact 0.6: Updike published children's books during his career.
Sentence 1:  Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.
  Fact 1.0: Seventeen of Updike's novels were bestsellers.
  Fact 1.1: More than 100 million copies of Updike's books are in print.

Retrieved items:
<item 1>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 1>

<item 2>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 2>

<item 3>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 3>

<item 4>
Tom Clancy's Net Force Explorers: Duel Identity
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 4>

<item 5>
Tom Clancy's Net Force Explorers: High Wire
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 5>

<item 6>
Tom Clancy's Net Force Explorers: Deathworld
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 6>

<item 7>
Joe McGinniss
Joseph R. McGinniss, Sr. (December 9, 1942 – March 10, 2014), known as Joe McGinniss, was an American non-fiction writer and novelist.

 The author of twelve books, he first came to prominence with the best-selling "The Selling of the President 1968" which described the marketing of
then-presidential candidate Richard Nixon.

 He is popularly known for his trilogy of bestselling true crime books — "Fatal Vision", "Blind Faith" and "Cruel Doubt" — which were adapted into TV miniseries
in the 1980s and 90s.

 His last book was "The Rogue: Searching for the Real Sarah Palin", an account of Sarah Palin, the former governor of Alaska who was the 2008 Republican
vice-presidential nominee.
</item 7>

<item 8>
Tom Clancy's Net Force Explorers: Gameprey
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 8>

<item 9>
Tom Clancy's Rainbow 6: Patriots
Tom Clancy's Rainbow 6: Patriots is a cancelled first-person shooter video game, part of the "Tom Clancy's Rainbow Six" series, announced on the cover of the
December 2011 issue of "Game Informer".

 It was to be published by Ubisoft, and was developed by the company's Montreal studio, with additional development by Ubisoft Toronto and Red Storm
Entertainment.

 Due to the death of Tom Clancy in October 2013, concern was raised that this game would become the last to bear his name.

 Ubisoft has since stated that they will continue putting Tom Clancy's name on future Tom Clancy titles out of respect for the late author.
</item 9>

<item 10>
Tom Clancy
Thomas Leo "Tom" Clancy Jr. (April 12, 1947 – October 1, 2013) was an American novelist best known for his technically detailed espionage and military-science
story lines set during and after the Cold War.

 Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.

 His name was also used on movie scripts written by ghost writers, nonfiction books on military subjects, and video games.

 He was a part-owner of the Baltimore Orioles and vice-chairman of their community activities and public affairs committees.
</item 10>
=== END ITEM id=57f8c2b015344517 ===

=== ITEM id=d756bf32f7397321 ===
Gold sentences and facts:
Sentence 0:  The S70 was essentially a facelifted 850 saloon.
  Fact 0.0: The S70 was essentially a facelifted 850 saloon.
Sentence 1: The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.
  Fact 1.0: The Volvo 850 is a compact executive car.
  Fact 1.1: The Volvo 850 was produced by the Swedish manufacturer Volvo Cars.
  Fact 1.2: The Volvo 850 was produced from 1991 to 1996.
Sentence 2:  The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.
  Fact 2.0: The Volvo 850 was the first front-wheel drive vehicle from Volvo to be sold in North America.
  Fact 2.1: The Volvo 850 was the first all-wheel drive Volvo.

Retrieved items:
<item 1>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 1>

<item 2>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 2>

<item 3>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 3>

<item 4>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 4>

<item 5>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 5>

<item 6>
Volvo 164
The Volvo 164 is a 4-door, 6-cylinder luxury sedan unveiled by Volvo at the Paris Motor Show early in October 1968 and first sold as a 1969 model.

 46,008 164s were built before the car was superseded by the 264 in 1975.

 The 164 was Volvo's first venture into the luxury segment since the end of PV 60 production in 1950, and was the first six-cylinder Volvo since the PV800 last
produced in 1958.
</item 6>

<item 7>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 7>

<item 8>
Volvo P2 platform
The Volvo P2 platform is a global full-size unibody automobile platform developed and produced by Volvo.

 It is designed for single wheelbases and is adaptable to front- or all wheel drive configurations.

 It was developed by the automaker before its 1999 acquisition by Ford Motor Company, and debuted with the 1998 Volvo S80.
</item 8>

<item 9>
Volvo’s 480 (1986–95), its first front-wheel-drive car, shared components with the Dutch-built 440 hatchback and 460 sedan (1987–96). The rear-wheel-drive 300
Series (1976–91) comprised 340/343/345 and 360 models. The luxury 164 (1968–75) was a six-cylinder sedan; 46,008 were built. Volvo’s compact transverse SI6
straight-six debuted in 200
</item 9>

<item 10>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 10>
=== END ITEM id=d756bf32f7397321 ===
