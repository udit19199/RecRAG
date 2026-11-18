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

This file has 11 items.

=== ITEM id=3b8c7532567827c3 ===
Gold sentences and facts:
Sentence 0: The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran on Monday, February 8, 1993, killing all 133 people on board both aircraft involved—an Iran Air Tours Tupolev Tu-154M, registered EP-ITD, and an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
  Fact 0.0: The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran.
  Fact 0.1: The collision occurred on Monday, February 8, 1993.
  Fact 0.2: The collision killed all 133 people on board both aircraft involved.
  Fact 0.3: One aircraft involved was an Iran Air Tours Tupolev Tu-154M, registered EP-ITD.
  Fact 0.4: The other aircraft involved was an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
Sentence 1: Iran Air Tours is an airline based in Tehran, Iran.
  Fact 1.0: Iran Air Tours is an airline.
  Fact 1.1: Iran Air Tours is based in Tehran, Iran.

Retrieved items:
<item 1>
1963 Aeroflot Tupolev Tu-124 Neva river ditching
The 1963 Aeroflot Tupolev Tu-124 Neva river ditching was a water landing by a Tupolev Tu-124 of the Soviet state airline "Aeroflot" (Moscow division).

 The aircraft took off from Tallinn-Ülemiste Airport (TLL) at 08:55 on 21 August with 45 passengers and 7 crew on board.

 The aircraft (registration number SSSR-45021) was built in 1962 and was scheduled to fly to Moscow–Vnukovo (VKO) under the command of 27-year-old captain
Victor Mostovoy.

 After takeoff the nose gear did not retract.

 Ground control diverted the flight to Leningrad (LED) – because of fog at Tallinn.
</item 1>

<item 2>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 2>

<item 3>
1955 Cincinnati mid-air collision
The 1955 Cincinnati mid-air collision occurred on January 12, 1955, when a Trans World Airlines Martin 2-0-2 on takeoff from Boone County Airport (now the
Cincinnati/Northern Kentucky International Airport) collided in mid-air with a privately owned Douglas DC-3 that had entered the airport's control space without
proper clearance.

 There were no survivors.
</item 3>

<item 4>
Iran Air Tours
Iran Air Tours is an airline based in Tehran, Iran.

 The airline mainly operates under the English name Iran Airtour (i.e. without the plural "s") as written in Persian (Persian: ایران ایر تور‎ ‎ ).

 This error is due to confusion between different pluralization rules in English and Persian.

 Iran Air Tours was established as a subsidiary of Iran Air and operates scheduled domestic services and international services in the Middle East as well as
Eastern Europe, as well as charter services.
</item 4>

<item 5>
Graph query row:
organization: Iran Air Tours
base_city: Tehran
</item 5>

<item 6>
Charkhi Dadri mid-air collision
The Charkhi Dadri mid-air collision occurred on 12 November 1996 over the village of Charkhi Dadri, to the west of New Delhi, India.

 The aircraft involved were a Saudi Arabian Airlines Boeing 747-100B en route from Delhi to Dhahran, Saudi Arabia, and a Kazakhstan Airlines Ilyushin Il-76 en
route from Chimkent, Kazakhstan, to Delhi.

 The crash killed all 349 people on board both planes, making it the world's deadliest mid-air collision, the deadliest aviation accident to occur in India, and
the third-deadliest aircraft accident in the history of aviation, behind only the Tenerife airport disaster and Japan Airlines Flight 123.
</item 6>

<item 7>
1993 Tehran mid-air collision
The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran on Monday, February 8, 1993, killing all 133 people on board both
aircraft involved—an Iran Air Tours Tupolev Tu-154M, registered EP-ITD, and an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
</item 7>

<item 8>
United Airlines Flight 736
United Airlines Flight 736 was a daily U.S. transcontinental passenger flight operated by United Airlines that crashed on April21, 1958, following a mid-air
collision.

 The aircraft assigned to Flight 736, a Douglas DC-7 airliner carrying 47 persons, was flying at cruise altitude above Clark County, Nevada, en route to a
stopover at Denver, Colorado, when it was struck by a United States Air Force fighter jet crewed by two pilots.

 The collision occurred at 8:30 a.m. in clear weather within a major commercial airway; both aircraft fell out of control from 21000 ft and crashed into
unpopulated desert terrain southwest of Las Vegas, Nevada.
</item 8>

<item 9>
1965 Carmel mid-air collision
The 1965 Carmel mid-air collision occurred on December 4, 1965, when Eastern Air Lines Flight 853 (N6218C), a Lockheed Super Constellation en route from Boston
Logan International Airport to Newark International Airport, collided in mid-air with Trans World Airlines Flight 42 (N748TW), a Boeing 707-131B en route from
San Francisco International Airport to John F. Kennedy International Airport, over Carmel, New York, United States.
</item 9>

<item 10>
1993 Auckland mid-air collision
The 1993 Auckland mid-air collision was an aircraft accident in New Zealand.

 It occurred on 26 November 1993, when two aircraft operated by Airwork, under contract to the New Zealand Police, collided and crashed in central Auckland.

 The mid-air collision of the Aérospatiale TwinStar helicopter and Piper Archer aeroplane resulted in the deaths of all four occupants – a civilian Airwork
pilot on each aircraft and two New Zealand Police officers on the helicopter.

 The accident occurred in daylight with excellent visibility, in uncontrolled airspace (class G), with both aircraft flying under visual flight rules.

 Both the helicopter and aeroplane were operated by Airwork (NZ), and working under contract to the New Zealand Police at the time of the accident.
</item 10>
=== END ITEM id=3b8c7532567827c3 ===

=== ITEM id=4120a67075639ba1 ===
Gold sentences and facts:
Sentence 0: Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.
  Fact 0.0: Underdog is a 2007 American film.
  Fact 0.1: Underdog is a family superhero comedy film.
  Fact 0.2: Underdog is based on the 1960s cartoon series of the same name.
Sentence 1: Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi, a Life in the Woods" by Austrian author Felix Salten.
  Fact 1.0: Bambi is a 1942 American animated film.
  Fact 1.1: Bambi was directed by David Hand.
  Fact 1.2: David Hand supervised a team of sequence directors on Bambi.
  Fact 1.3: Bambi was produced by Walt Disney.
  Fact 1.4: Bambi is based on the book 'Bambi, a Life in the Woods'.
  Fact 1.5: The book 'Bambi, a Life in the Woods' was written by Austrian author Felix Salten.

Retrieved items:
<item 1>
Bambi
Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi,
a Life in the Woods" by Austrian author Felix Salten.

 The film was released by RKO Radio Pictures on August 13, 1942, and is the fifth Disney animated feature film.
</item 1>

<item 2>
Bosko the Musketeer
Bosko the Musketeer is an American animated short film.

 It is a "Looney Tunes" cartoon, featuring Bosko, the first star of the series.

 It was released on August 12, 1933, although some sources note September 16 as a date; this is problematic, as that would imply that the last films featuring
Bosko as the star of Warner Bros. cartoons were released after the first film featuring Buddy, the second star of the series.

 It was, like most "Looney Tunes" of the time, directed by Hugh Harman; Frank Marsales was the musical director.
</item 2>

<item 3>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 3>

<item 4>
7 Seconds (film)
7 Seconds is a 2005 American crime film directed by Simon Fellows, starring Wesley Snipes and Tamzin Outhwaite.

 The film was released on Direct-to-DVD in the United States on June 28, 2005.

 The title refers to the timers at the beginning of the film, which are set at 00:07 (7 seconds).
</item 4>

<item 5>
Roadside Romeo
Roadside Romeo is a 2008 3D Indian-American computer animated romantic musical comedy family film written and directed by Jugal Hansraj and produced by Aditya
Chopra and Yash Chopra of Yash Raj Films and distributed by Walt Disney Studios Motion Pictures in United States, United Kingdom and India.

 It was released on 24 October 2008 in the United States and India.

 An Arabic dub was released in Kuwait on October 23.

 This was the second Bollywood movie to receive a North American release by a Hollywood studio, following Sony Pictures' "Saawariya" (2007).

 The title character is a dog living in Mumbai, as voiced by Saif Ali Khan; his girlfriend, Laila, is voiced by Kareena Kapoor.

 This was the first voice-over in an animated production for both actors.

 "Roadside Romeo" was also Hansraj's directorial debut.

 "Roadside Romeo" received generally negative reviews from critics, with most of the criticism focused on the film's script, predictable plot and overuse of
cliches.
</item 5>

<item 6>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 6>

<item 7>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 7>

<item 8>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 8>

<item 9>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 9>

<item 10>
Anthem for the Underdog
Anthem for the Underdog is the third album by the rock band 12 Stones.

 It was released on August 14, 2007.

 The first single "Lie to Me", was made available on 12 Stones' MySpace and official website along with the second single "Anthem For The Underdog", which was
used in the movie "Never Back Down".

 Both singles charted in the top 30 on the Mainstream Rock Tracks chart, while "It Was You" charted in the top 10 on Christian Rock charts.

 The third single, "Adrenaline", was the theme song for the Met-Rx World's Strongest Man competition in 2007.

 "Anthem for the Underdog" debuted at No. 53 on the "Billboard" 200, and stayed on the chart for four weeks.
</item 10>
=== END ITEM id=4120a67075639ba1 ===

=== ITEM id=82149a900f4f35d0 ===
Gold sentences and facts:
Sentence 0:  He earned 1 cap under Jack Charlton for the Irish national team.
  Fact 0.0: The player earned 1 cap for the Irish national team.
  Fact 0.1: The player earned that cap under Jack Charlton.
Sentence 1:  He was part of the England team that won the 1966 World Cup.
  Fact 1.0: The player was part of the England team that won the 1966 World Cup.
  Fact 1.1: The England team won the 1966 World Cup.

Retrieved items:
<item 1>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 1>

<item 2>
Walter Ward (footballer)
Walter Ward (1869 – after 1890) was an English professional footballer born in Birmingham who played in the Football Alliance for Small Heath.

 During the 1890–91 season, Ward deputised for regular goalkeeper Chris Charsley when Charsley's duties as a serving police officer demanded his absence from
the side.
</item 2>

<item 3>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 3>

<item 4>
1929–30 British Home Championship
The 1929-30 British Home Championship was an edition of the annual international football tournament played between the British Home Nations.

 1930 was the year in which the tournament finally gained a serious rival as the premier international football competition, with the inception of the 1930 FIFA
World Cup, held in Uruguay.

 The Home Nations were not however members of FIFA due to disputes over the growing professionalism in continental and South American football.

 As a result, they were not able to attend and indicated that even if they were invited they would have no interest in attending, deeming foreign opposition too
weak for serious contest.

 It is interesting to speculate what would have happened had the Home Nations entered the tournament, especially as the tide of world football was changing
against Britain.

 The England team, which dominated the 1930 championship, had lost to Spain the year before in the first defeat by a foreign football team, and in the same year
they only managed draws with Germany and Austria.

 The Scottish side, which had won most of the previous ten championships, was likewise unprepared, only playing its first game outside the British Isles in
1929, and being heavily defeated on tour in 1931 by both the Austrians and the Italians.
</item 4>

<item 5>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 5>

<item 6>
World Soccer (magazine)
World Soccer is an English language football magazine published by IPC Media.

 The magazine was established in 1960.

 It specialises in the international football scene.

 Its regular contributing writers include Brian Glanville, Sid Lowe and Tim Vickery.

 "World Soccer" is a member of the European Sports Magazines (ESM), an umbrella group of similar magazines printed in other languages.

 Other members include "A Bola", "Don Balón", "Kicker", "La Gazzetta dello Sport" and "Sport Express".

 The members of this group elect a European "Team of the Month" and a European "Team of the Year".
</item 6>

<item 7>
Barry Rose
Barry Michael Rose OBE, FRAM, FRSCM, Hon.D.Mus, Hon.FGCM, Hon FRCO, M.Univ., (Surrey), born 24 May 1934) is a choir trainer and organist.

 He is best known for founding the choir and the pattern of daily sung Worship at the new Guildford Cathedral in 1961, and later, on 29 July 1981, directing the
music at the wedding of Charles, Prince of Wales and Diana, Princess of Wales formerly Lady Diana Frances Spencer, at St Paul's Cathedral in London.
</item 7>

<item 8>
2002 Commonwealth Games
The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25
July to 4 August 2002.

 The 2002 Games were to be hosted in the United Kingdom to coincide with the Golden Jubilee of Elizabeth II, head of the Commonwealth, and Manchester was
selected for the 2002 Games ahead of London.

 The XVII Commonwealth Games was, prior to the 2012 Summer Olympics, the largest multi-sport event ever to be held in the UK, eclipsing the London 1948 Summer
Olympics in numbers of teams and athletes participating.

 In terms of sports and events, the 2002 Games were the largest Commonwealth Games in history featuring 281 events across 17 sports.
</item 8>

<item 9>
Barry Murphy (footballer, born 1959)
Barry Murphy (born 1 April 1959 in Dublin) was an Irish soccer player during the 1980s.

 He earned 1 cap under Jack Charlton for the Irish national team.
</item 9>

<item 10>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 10>
=== END ITEM id=82149a900f4f35d0 ===

=== ITEM id=fa1cb121172b6a31 ===
Gold sentences and facts:
Sentence 0: Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with punctate stigmas on the female flowers.
  Fact 0.0: Lithocarpus is a genus in the beech family Fagaceae.
  Fact 0.1: Lithocarpus differs from "Quercus" in the erect spikes of insect-pollinated male flowers.
  Fact 0.2: Lithocarpus differs from "Quercus" in the short styles with punctate stigmas on the female flowers.
Sentence 1:  It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.
  Fact 1.0: Lithocarpus contains 17 species of shrubs and small trees.
  Fact 1.1: The Lithocarpus species are native from southern Florida to Mexico and South America.

Retrieved items:
<item 1>
Lithocarpus kostermansii
Lithocarpus kostermansii is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 1>

<item 2>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 2>

<item 3>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 3>

<item 4>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 4>

<item 5>
Dregea sinensis
Dregea sinensis is a species of plant in the Apocynaceae family that is native to China.
</item 5>

<item 6>
Lithocarpus platycarpus
Lithocarpus platycarpus is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.

.
</item 6>

<item 7>
Lithocarpus glaber
Lithocarpus glaber, the Japanese oak, is a tree species in the genus "Lithocarpus" found in Japan, China and Taïwan.
</item 7>

<item 8>
Lithocarpus crassinervius
Lithocarpus crassinervius is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 8>

<item 9>
Graph query row:
lithocarpus_species: Lithocarpus glaber
lithocarpus_relationship: LOCATED_IN
shared_entity: China
duranta_relationship: LOCATED_IN
duranta_species: Duranta erecta
</item 9>

<item 10>
Duranta erecta
Duranta erecta is a species of flowering shrub in the verbena family Verbenaceae, native from Mexico to South America and the Caribbean.

 It is widely cultivated as an ornamental plant in tropical and subtropical gardens throughout the world, and has become naturalized in many places.

 It is considered an invasive species in Australia, China, South Africa and on several Pacific Islands.

 Common names include golden dewdrop, pigeon berry, and skyflower.

 In Mexico, the native Nahuatl name for the plant is xcambocoché.

 In Tonga it is known as mavaetangi (tears of departure).
</item 10>
=== END ITEM id=fa1cb121172b6a31 ===

=== ITEM id=fb551acf1313ef41 ===
Gold sentences and facts:
Sentence 0:  He is most well known for providing the singing voice of Tony (played onscreen by Richard Beymer) in the 1961 film musical "West Side Story".
  Fact 0.0: This person is most well known for providing the singing voice of Tony in the 1961 film musical "West Side Story".
  Fact 0.1: Richard Beymer played Tony onscreen in the 1961 film musical "West Side Story".
Sentence 1:  The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the most wins for a movie musical.
  Fact 1.0: West Side Story was nominated for 11 Academy Awards.
  Fact 1.1: West Side Story won 10 Academy Awards.
  Fact 1.2: West Side Story won Best Picture.
  Fact 1.3: West Side Story won a special award for Robbins.
  Fact 1.4: West Side Story became the record holder for the most wins for a movie musical.

Retrieved items:
<item 1>
The Skydivers
The Skydivers is a 1963 film produced by Anthony Cardoza and written and directed by Coleman Francis, who together also made The Beast of Yucca Flats.

 It stars actress Kevin Casey as Beth, along with, also from "The Beast of Yucca Flats", Eric Tomlin as Joe, Anthony Cardoza as Harry, and Marcia Knight as
Suzy.

 The film contains performances by influential Nashville guitarist Jimmy Bryant.
</item 1>

<item 2>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 2>

<item 3>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 3>

<item 4>
Jimmy Bryant (singer)
James Howard Bryant (born June 2, 1929) is a singer, arranger and composer.

 He is most well known for providing the singing voice of Tony (played onscreen by Richard Beymer) in the 1961 film musical "West Side Story".

 While he received no screen credit, he states that Beymer was "a nice guy, and every time he did an interview he would mention my name."

 He also sang for James Fox in the 1967 film musical "Thoroughly Modern Millie", and sang in "The Telephone Hour" number in "Bye Bye Birdie".

 He also sang in the group that performed the theme song of the TV series "Batman".
</item 4>

<item 5>
Geoffrey Rush
Geoffrey Roy Rush {'1': ", '2': ", '3': ", '4': "} (born 6 July 1951) is an Australian actor and film producer.

 Rush is the youngest amongst the few people who have won the "Triple Crown of Acting": the Academy Award, the Primetime Emmy Award, and the Tony Award.

 He has won one Academy Award for acting (from four nominations), three British Academy Film Awards (from five nominations), two Golden Globe Awards and four
Screen Actors Guild Awards. Rush is the founding President of the Australian Academy of Cinema and Television Arts and was named the 2012 Australian of the
Year.

 He is also the first actor to win the Academy Award, BAFTA Award, Critics' Choice Movie Award, Golden Globe Award, and Screen Actors Guild Award for a single
performance in film for his performance in "Shine" (1996).
</item 5>

<item 6>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 6>

<item 7>
West Side Story (film)
West Side Story is a 1961 American musical romantic drama film directed by Robert Wise and Jerome Robbins.

 The film is an adaptation of the 1957 Broadway musical of the same name, which in turn was inspired by William Shakespeare's play "Romeo and Juliet".

 It stars Natalie Wood, Richard Beymer, Russ Tamblyn, Rita Moreno, and George Chakiris, and was photographed by Daniel L. Fapp, A.S.C., in Super Panavision 70.

 Released on October 18, 1961 through United Artists, the film received high praise from critics and viewers, and became the second highest grossing film of the
year in the United States.

 The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the
most wins for a movie musical.
</item 7>

<item 8>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 8>

<item 9>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 9>

<item 10>
Speedy West
Wesley Webb West (January 25, 1924 – November 15, 2003), better known as Speedy West, was an American pedal steel guitarist and record producer.

 He frequently played with Jimmy Bryant, both in their own duo and as part of the regular Capitol Records backing band for Tennessee Ernie Ford and many others.

 He also played on Loretta Lynn's first single.
</item 10>
=== END ITEM id=fb551acf1313ef41 ===

=== ITEM id=2c29f7635305941f ===
Gold sentences and facts:
Sentence 0: "I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.
  Fact 0.0: "I'll Keep the Lovelight Burning" is a popular song.
  Fact 0.1: "I'll Keep the Lovelight Burning" was written in 1942.
  Fact 0.2: "I'll Keep the Lovelight Burning" was written by Harry Tobias.
  Fact 0.3: "I'll Keep the Lovelight Burning" was written by Nick Kenny.
  Fact 0.4: "I'll Keep the Lovelight Burning" was written by Harold Levey.
  Fact 0.5: "I'll Keep the Lovelight Burning" was popularized by Patti Page.
  Fact 0.6: Patti Page popularized "I'll Keep the Lovelight Burning" in 1949.
Sentence 1: Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country music.
  Fact 1.0: Clara Ann Fowler's date range is November 8, 1927 to January 1, 2013.
  Fact 1.1: Clara Ann Fowler is known by the professional name Patti Page.
  Fact 1.2: Patti Page was an American singer.
  Fact 1.3: Patti Page was a singer of traditional pop music.
  Fact 1.4: Patti Page was a singer of country music.

Retrieved items:
<item 1>
For You (Ricky Nelson song)
"For You" is a song written by Joe Burke and Al Dubin in 1930.

 Dubin wrote the songs for the Broadway show "42nd Street".

 The Glen Gray Orchestra recorded it with Kenny Sargent doing the vocals.

 Perry Como recorded it in November 1947, releasing the song in 1948.

 Dean Martin recorded it for an album in 1957.

 Timi Yuro also recorded the song.

 It was released on her LP "Timi Yuro" (Liberty LRP-3208 (mono) and Liberty LST-7208 (stereo)) in 1961.
</item 1>

<item 2>
The entries cover songs and recordings spanning R&B, rock, pop, and folk. Highlights include Johnny Otis’s 1958 “Willie and the Hand Jive” (US #9), later
covered by Eric Clapton (#26) and George Thorogood; Van Halen’s “Finish What Ya Started,” co-written overnight by Eddie Van Halen and Sammy Hagar; and Johnny
Otis’s “Every Beat of My Heart,” later a major hit for the Pips.
</item 2>

<item 3>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 3>

<item 4>
Keep This Fire Burning
"Keep This Fire Burning" is a song by Swedish pop singer Robyn, released as the first single from her third album "Don't Stop the Music".

 It was released in Sweden on September 21, 2002, where it became her highest charting single since 1995's "Do You Really Want Me (Show Respect)".

 The song was also released as a single in Australia under the name "By Your Side", due to the Australian bush fires which were happening at the time.

 In 2008, a re-recorded version of the song appeared on the special edition of Robyn's eponymous album.
</item 4>

<item 5>
You Keep Me Hangin' On
"You Keep Me Hangin' On" is a 1966 song written and composed by Holland–Dozier–Holland.

 It first became a popular "Billboard" Hot 100 number one hit for the American Motown group The Supremes in late 1966.

 The rock band Vanilla Fudge covered the song a year later and had a top ten hit with their version.

 British pop singer Kim Wilde covered "You Keep Me Hangin' On" in 1986, bumping it back to number one on the "Billboard" Hot 100 in June 1987.

 The single reached number one by two different musical acts in America.

 In the first 32 years of the "Billboard" Hot 100 rock era, “You Keep Me Hangin' On” became one of only six songs to achieve this feat.

 In 1996, country music singer Reba McEntire's version reached number 2 on the US "Billboard" Hot Dance Club Play chart.
</item 5>

<item 6>
Robson & Jerome’s 1995 debut topped the UK chart, selling 2.04 million copies; its 42-day record for reaching two million lasted until Adele’s *25*. Forbes’
Beatles tribute won Sweden’s 1977 selection but finished last at Eurovision. Dylan’s 1978 “Baby, Stop Crying” reached UK #13. Jobim’s “Corcovado” became “Quiet
Nights of Quiet Stars”; Andy Williams and Kitty Kallen recorded it. “
</item 6>

<item 7>
Live in Old Smokey
Live in Old Smokey is a live CD by English singer/songwriter Linda Lewis recorded at Ronnie Scott's in London and issued in 2006.
</item 7>

<item 8>
I'll Keep the Lovelight Burning
"I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.

 Louis Armstrong also covered the song in 1949.
</item 8>

<item 9>
Jimmy O'Neill (DJ)
James Franklin O'Neill (January 8, 1940 – January 11, 2013) was an American DJ and broadcaster who hosted the ABC television show "Shindig!

" from 1964-1966.

 O'Neill was owner of Pandora's Box, an influential Sunset Strip music venue in West Hollywood, California that was the center of the 1966 Sunset Strip curfew
riots.
</item 9>

<item 10>
Patti Page
Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country
music.

 She was the top-charting female vocalist and best-selling female artist of the 1950s, selling over 100 million records during a six-decade long career.

 She was often introduced as "the Singin' Rage, Miss Patti Page".

 New York WNEW disc-jockey William B. Williams introduced her as "A Page in my life called Patti".
</item 10>
=== END ITEM id=2c29f7635305941f ===

=== ITEM id=1529d3d57d1918db ===
Gold sentences and facts:
Sentence 0: Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.
  Fact 0.0: Nelson Algren was born March 28, 1909.
  Fact 0.1: Nelson Algren died May 9, 1981.
  Fact 0.2: Nelson Algren was an American writer.
Sentence 1: Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
  Fact 1.0: Nathanael West was born Nathan Weinstein.
  Fact 1.1: Nathanael West was born October 17, 1903.
  Fact 1.2: Nathanael West died December 22, 1940.
  Fact 1.3: Nathanael West was an American author.
  Fact 1.4: Nathanael West was an American screenwriter.
  Fact 1.5: Nathanael West was an American satirist.

Retrieved items:
<item 1>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 1>

<item 2>
Romeo Nelson
Iromeio "Romeo" Nelson (March 12, 1902 – May 17, 1974) was an American boogie woogie pianist whose recordings from 1929 are regarded as some of the finest, and
certainly the fastest, boogie woogie showpieces on record.
</item 2>

<item 3>
Andrew Glaze
Andrew Glaze (April 21, 1920 – February 7, 2016) was an American poet, playwright and novelist.

 Much of Glaze's poetry reflects his coming of age in the South, and his eventual return there.

 He also lived and wrote in New York City for 31 years.

 In New York City he became part of a circle of poets that included Oscar Williams, Norman Rosten, John Ciardi and William Packard.
</item 3>

<item 4>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 4>

<item 5>
Nathanael West
Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
</item 5>

<item 6>
The Man with the Golden Arm (novel)
The Man with the Golden Arm is a novel by Nelson Algren, published by Doubleday in November 1949.

 One of the seminal novels of post-World War II American letters, "The Man with the Golden Arm" is widely considered Algren's greatest and most enduring work.

 It won the National Book Award in 1950.
</item 6>

<item 7>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 7>

<item 8>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 8>

<item 9>
A Walk on the Wild Side
A Walk on the Wild Side is a 1956 novel by Nelson Algren, most often quoted as the source for Algren's "three rules of life": "Never play cards with a man
called Doc.

 Never eat at a place called Mom's.

 Never sleep with a woman whose troubles are worse than your own."
</item 9>

<item 10>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 10>
=== END ITEM id=1529d3d57d1918db ===

=== ITEM id=2373b047721e7970 ===
Gold sentences and facts:
Sentence 0: Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in 1995.
  Fact 0.0: Haunted is the second album by American singer-songwriter Poe.
  Fact 0.1: Haunted was released in 2000.
  Fact 0.2: Haunted was released after a five-year hiatus from Poe's debut album 'Hello', which came out in 1995.
Sentence 1:  The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
  Fact 1.0: Haunted was self-produced.
  Fact 1.1: Haunted was created as a tribute to Poe's father.
  Fact 1.2: Haunted was created as a counterpart to the novel 'House of Leaves' by Poe's brother Mark Z. Danielewski.
Sentence 2:  A bestseller, it has been translated into a number of languages, and is followed by a companion piece, "The Whalestoe Letters".
  Fact 2.0: Haunted is a bestseller.
  Fact 2.1: Haunted has been translated into a number of languages.
  Fact 2.2: Haunted is followed by a companion piece, 'The Whalestoe Letters'.

Retrieved items:
<item 1>
Uncle Wiggily in Connecticut
"Uncle Wiggily in Connecticut" is a short story by J. D. Salinger, which appears in his collection "Nine Stories".

 It was originally published in the March 20, 1948 issue of "The New Yorker".
</item 1>

<item 2>
James Poe
James Poe (October 4, 1921 – January 24, 1980) was an American film and television screenwriter.

 He is best known for his work on the movies "Around the World in 80 Days" for which he jointly won an Academy Award in 1956, "Cat on a Hot Tin Roof", "Summer
and Smoke", "Lilies of the Field", and "They Shoot Horses, Don't They?

".
</item 2>

<item 3>
The Pig, or Václav Havel's Hunt for a Pig
The Pig, or Václav Havel's Hunt for a Pig is the final work by Václav Havel, co-authored by Vladimír Morávek.

 The English translation is by Edward Einhorn .

 Originally a short dialogue from 1987 (entitled simply "The Pig") and printed in a samizdat, the piece is a comic (and true) story of Václav Havel’s efforts to
hold a pig roast for his friends.
</item 3>

<item 4>
The Father-Thing (collection)
The Father-Thing is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Gollancz in 1989 and reprints Volume III of "The Collected Stories of Philip K. Dick".

 It had not previously been published as a stand-alone volume.

 Many of the stories had originally appeared in the magazines "If", "Science Fiction Adventures", "Science Fiction Stories", "Orbit", "Fantasy and Science
Fiction", "Imagination", "Future", "Galaxy Science Fiction", "Beyond Fantasy Fiction", "Satellite", "Science Fiction Quarterly" and "Imaginative Tales".
</item 4>

<item 5>
The Frost King
"The Frost King" was a short story about King Jack Frost written by 11-year-old Helen Keller.

 Keller's teacher, Anne Sullivan, had mentioned that the autumn leaves were "painted ruby, emerald, gold, crimson, and brown," and Keller, by her own account,
imagined fairies doing the work.

 Keller wrote a story about how a cask of jewels, being transported by fairy servants, had melted in the sun and covered the leaves.

 As a birthday gift, Keller sent the story to Michael Anagnos, head of the Perkins School for the Blind, who published the story in "The Mentor," the Perkins
alumni magazine.

 It was picked up by "The Goodson Gazette", a journal on deaf-blind education, based in Virginia.
</item 5>

<item 6>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 6>

<item 7>
Haunted (Poe album)
Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in
1995.

 The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
</item 7>

<item 8>
The Spectre Knight
The Spectre Knight is a one-act "fanciful operetta" with a libretto by James Albery and music by Alfred Cellier.

 It was first performed on 9 February 1878 at the Opera Comique by the Comedy Opera Company as a companion piece to "The Sorcerer".

 The piece continued to run until 23 March 1878 and was revived by the company from 28 May 1878 to 10 August 1878 as a companion piece to "H.M.S. Pinafore".

 The piece had a run in New York in 1880 and was toured in Britain and America.
</item 8>

<item 9>
Brothers in Arms (Bujold novel)
Brothers in Arms is a science fiction novel by American writer Lois McMaster Bujold, part of the Vorkosigan Saga.

 It was the fifth book published in the series, and is the twelfth story, including novellas, in the internal chronology of the series.

 "Brothers in Arms" was first published by Baen Books in January 1989, and is included in the 2002 omnibus "Miles Errant".
</item 9>

<item 10>
A Predicament
"A Predicament" is a humorous short story by Edgar Allan Poe, usually combined with its companion piece "How to Write a Blackwood Article."

 It was originally titled "The Scythe of Time".

 The paired stories parody the Gothic sensation tale, popular in England and America since the early 19th century.
</item 10>
=== END ITEM id=2373b047721e7970 ===

=== ITEM id=98c1b98969743360 ===
Gold sentences and facts:
Sentence 0: Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from 1997-1998.
  Fact 0.0: Robert Suettinger was the national intelligence officer for East Asia at the National Intelligence Council (NIC).
  Fact 0.1: Robert Suettinger held the role of national intelligence officer for East Asia for United States President Bill Clinton.
  Fact 0.2: Robert Suettinger held the role from 1997 to 1998.
Sentence 1: William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States from 1993 to 2001.
  Fact 1.0: William Jefferson Clinton was born William Jefferson Blythe III.
  Fact 1.1: William Jefferson Clinton was born on August 19, 1946.
  Fact 1.2: William Jefferson Clinton is an American politician.
  Fact 1.3: William Jefferson Clinton served as the 42nd President of the United States.
  Fact 1.4: William Jefferson Clinton served as President from 1993 to 2001.
Sentence 2:  Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.
  Fact 2.0: Before the presidency, William Jefferson Clinton was Governor of Arkansas.
  Fact 2.1: William Jefferson Clinton was Governor of Arkansas during two separate terms.
  Fact 2.2: William Jefferson Clinton's first term as Governor of Arkansas was from 1979 to 1981.
  Fact 2.3: William Jefferson Clinton's second term as Governor of Arkansas was from 1983 to 1992.

Retrieved items:
<item 1>
National Intelligence Estimate
National Intelligence Estimates (NIEs) are United States federal government documents that are the authoritative assessment of the Director of National
Intelligence (DNI) on intelligence related to a particular national security issue.

 NIEs are produced by the National Intelligence Council and express the coordinated judgments of the United States Intelligence Community, the group of 16 U.S.
intelligence agencies.

 NIEs are classified documents prepared for policymakers.
</item 1>

<item 2>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 2>

<item 3>
Steven K. Galson
Steven Kenneth Galson (born 1956) is an American public health physician.

 He is currently Senior Vice President for Global Regulatory Affairs at Amgen, the S. California-based innovative global biopharmaceutical company.

 He is also Professor-at-Large at the Keck Graduate Institute for Applied Life Sciences in Clarmont, California.

 He is a retired rear admiral in the United States Public Health Service Commissioned Corps and public health administrator who served as the acting Surgeon
General of the United States from October 1, 2007 – October 1, 2009.

 He served concurrently as acting Assistant Secretary for Health from January 22, 2009 to June 25, 2009, and as the Deputy Director and Director of the Center
for Drug Evaluation and Research (CDER) at the Food and Drug Administration from 2001 to 2007.

 As the Acting Surgeon General, he was the commander of the United States Public Health Service Commissioned Corps and, while serving as the Assistant Secretary
for Health, was the operational head of the Public Health Service.
</item 3>

<item 4>
National Intelligence Cross
The National Intelligence Cross is a decoration of the United States Intelligence Community (IC) awarded under the National Intelligence Awards (NIA) Program by
the Office of the Director of National Intelligence (ODNI).

 It is the highest award presented by the Office of the Director of National Intelligence.

 It is equivalent of the Central Intelligence Agency's Distinguished Intelligence Cross.
</item 4>

<item 5>
Robert Suettinger
Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from
1997-1998.

 While there, Suettinger oversaw the preparation of national intelligence estimates for the director of the Central Intelligence Agency.

 His areas of specialty included the People's Republic of China and the North Korean nuclear weapons program.

 After working in the Clinton administration, Suettinger has been a senior analyst for the Brookings Institution where he wrote the book "Beyond Tiananmen – The
Politics of U.S.-China Relations, 1989-2000".

 He also is a senior advisor at The Stimson Center.
</item 5>

<item 6>
National Intelligence Board
The National Intelligence Board (formerly the National Foreign Intelligence Board) is a body of senior U.S. intelligence community leaders led by the Director
of National Intelligence (DNI).

 The Board is tasked with reviewing and approving National Intelligence Estimates (NIEs).
</item 6>

<item 7>
1997 Alabama Crimson Tide football team
The 1997 Alabama Crimson Tide football team represented the University of Alabama in the college football season of 1997–98.

 The team's head coach was Mike DuBose, who was entering his first year at Alabama.

 They played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in Birmingham, Alabama and competed in the West Division of the
Southeastern Conference.

 Alabama would finish with a record of 4–7 record in suffering the program's first losing season since the 1984 season.

 The loss against Kentucky marked Alabama's first ever overtime game, as overtime rules for college football had gone into effect the previous season.
</item 7>

<item 8>
Lawrence Eagleburger
Lawrence Sidney Eagleburger (August 1, 1930 – June 4, 2011) was an American statesman and career diplomat, who served briefly as the Secretary of State under
President George H. W. Bush.

 Previously, he had served in lesser capacities under Presidents Richard Nixon, Jimmy Carter, and Ronald Reagan, and as Deputy Secretary of State under George
H. W. Bush.
</item 8>

<item 9>
Yannis Pappas
Yannis Pappas is a Greek-American comedian born and raised in Park Slope Brooklyn, New York.

 Both of his parents, Anna and Chris, practiced law.

 He is the youngest of three children.

 He has two older brothers, Peter and Nikos Pappas.

 His brother Peter is also a lawyer who worked for the Clinton and Obama administrations.

 Yannis attended college at the American University in Washington, D.C. where he studied American Studies and History.

 Pappas previously also worked in the social work field where he participated in 9/11 disaster relief for two and a half years and worked with the mentally ill
and people dealing with homelessness for three years with Lutheran Social Services.
</item 9>

<item 10>
Director of the Defense Intelligence Agency
The Director of the Defense Intelligence Agency is an intelligence officer who, upon nomination by the President and confirmation by the Senate, serves as the
nation's highest-ranking military intelligence officer.

 As the chief of the Defense Intelligence Agency, the Director is the primary intelligence adviser to the Secretary of Defense and the Chairman of the Joint
Chiefs of Staff and also answers to the Director of National Intelligence through the civilian Under Secretary of Defense for Intelligence.

 The Director is also the Commander of the Joint Functional Component Command for Intelligence, Surveillance and Reconnaissance, a subordinate command of United
States Strategic Command.

 Additionally, he chairs the Military Intelligence Board, which coordinates activities of the entire defense intelligence community.
</item 10>
=== END ITEM id=98c1b98969743360 ===

=== ITEM id=6b41aeb16e9f52c5 ===
Gold sentences and facts:
Sentence 0:  The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.
  Fact 0.0: The pool lies directly south of Lichfield Cathedral.
  Fact 0.1: The pool has historically been important to the defence of the Cathedral Close.
Sentence 1:  The Diocese of Lichfield covers all of Staffordshire, much of Shropshire and part of the Black Country and West Midlands.
  Fact 1.0: The Diocese of Lichfield covers all of Staffordshire.
  Fact 1.1: The Diocese of Lichfield covers much of Shropshire.
  Fact 1.2: The Diocese of Lichfield covers part of the Black Country.
  Fact 1.3: The Diocese of Lichfield covers part of the West Midlands.

Retrieved items:
<item 1>
Tipton
Tipton is a town in the borough of Sandwell, West Midlands, England, with a population of around 38,777 at the 2011 UK Census.

 Tipton is located about halfway between Birmingham and Wolverhampton.

 It is a part of the West Midlands conurbation and is a part of the Black Country.
</item 1>

<item 2>
Graph query row:
place: Minster Pool
labels: ['__KGBuilder__', '__Entity__', 'Thing']
connections: [{'country': None, 'held_on': None, 'kind': 'city', 'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Lichfield', 'relationship':
'LOCATED_IN'}]
regional_paths: [{'relationships': ['FROM_CHUNK', 'FROM_CHUNK', 'PART_OF', 'LOCATED_IN'], 'entities': ['Minster Pool', None, 'United Kingdom', 'England', 'West
Midlands']}, {'relationships': ['LOCATED_IN', 'LOCATED_IN', 'PART_OF', 'LOCATED_IN'], 'entities': ['Minster Pool', 'Lichfield', 'Staffordshire', 'Brierley
Hill', 'West Midlands']}, {'relationships': ['FROM_CHUNK', 'FROM_CHUNK', 'PART_OF', 'LOCATED_IN'], 'entities': ['Minster Pool', None, 'Staffordshire', 'Brierley
Hill', 'West Midlands']}, {'relationships': ['LOCATED_IN', 'LOCATED_IN', 'FROM_CHUNK', 'FROM_CHUNK'], 'entities': ['Minster Pool', 'Lichfield', 'Staffordshire',
None, 'West Midlands']}, {'relationships': ['FROM_CHUNK', 'FROM_CHUNK', 'FROM_CHUNK', 'FROM_CHUNK'], 'entities': ['Minster Pool', None, 'Staffordshire', None,
'West Midlands']}, {'relationships': ['LOCATED_IN', 'LOCATED_IN', 'FROM_CHUNK', 'FROM_CHUNK'], 'entities': ['Minster Pool', 'Lichfield', 'Staffordshire', None,
'Black Country']}, {'relationships': ['FROM_CHUNK', 'FROM_CHUNK', 'FROM_CHUNK', 'FROM_CHUNK'], 'entities': ['Minster Pool', None, 'Staffordshire', None, 'Black
Country']}, {'relationships': ['LOCATED_IN', 'LOCATED_IN', 'PART_OF', 'PART_OF'], 'entities': ['Minster Pool', 'Lichfield', 'Staffordshire', 'Brierley Hill',
'Black Country']}, {'relationships': ['FROM_CHUNK', 'FROM_CHUNK', 'PART_OF', 'PART_OF'], 'entities': ['Minster Pool', None, 'Staffordshire', 'Brierley Hill',
'Black Country']}]
</item 2>

<item 3>
List of railway stations in the West Midlands
This is a list of railway stations within the West Midlands, a metropolitan county in central England which includes the cities of Birmingham, Coventry and
Wolverhampton.

 It includes all railway stations in the West Midlands that currently have regular timetabled train services, as well as certain stations outside the county
which are within the area supported by Transport for West Midlands (TfWM), formerly known as Centro.

 Transport within the West Midlands is subsidised by TfWM, who since 2006 have used the brand name Network West Midlands to demonstrate the 'joined-up' nature
of the regions bus and rail networks.
</item 3>

<item 4>
Minster Pool
Minster Pool is a reservoir located between Bird Street and Dam Street in the heart of the city of Lichfield, Staffordshire in the United Kingdom.

 The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.

 The pool was originally formed in the 11th century when a boggy stream was dammed at its eastern end to drive a mill on Dam Street.

 The pool was used as a mill pond and fishery until 1856 when the mill was demolished; it has since been retained for public amenity.
</item 4>

<item 5>
Black Country Living Museum
The Black Country Living Museum (formerly The Black Country Museum) is an open-air museum of rebuilt historic buildings in Dudley in the West Midlands of
England.

 It is located in the centre of the Black Country, 10 miles west of Birmingham.

 The museum occupies 105000 m2 of former industrial land partly reclaimed from a former railway goods yard, disused lime kilns, canal arm and former coal pits.
</item 5>

<item 6>
Black Country Rangers F.C.
Black Country Rangers Football Club is a football club based in Rowley Regis in the West Midlands.

 They are currently members of the West Midlands (Regional) League Premier Division and play at the Beeches in Cradley, groundsharing with Cradley Town.
</item 6>

<item 7>
Walsall
Walsall ( ) is an industrial town in the West Midlands of England.

 It is located 8 miles north-west of the City of Birmingham and 6 miles east of the City of Wolverhampton.

 Historically a part of Staffordshire, Walsall is a component area of the West Midlands conurbation, and part of the Black Country.
</item 7>

<item 8>
Dudley Zoo
Dudley Zoological Gardens is a 40 acre zoo located within the grounds of Dudley Castle in the town of Dudley, in the Black Country region of the West Midlands,
England.

 The Zoo opened to the public on 18 May 1937.

 Dudley Zoo is owned and operated by Dudley and West Midlands Zoological Society, founded in 1935 and a registered charity.
</item 8>

<item 9>
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
</item 9>

<item 10>
Brierley Hill
Brierley Hill is a small town and electoral ward of the Dudley Metropolitan Borough, in the West Midlands of England, and is situated approximately 2.5 miles
south of central Dudley and 2 miles north of Stourbridge.

 Part of the Black Country, and in a heavily industrialised area of the Dudley Borough, it has a population of 13,935 at the 2011 census, and is best known for
glass and steel manufacturing, although the industry has declined considerably since the 1970s.

 One of the largest factories in the area was the Round Oak Steelworks, which was closed down and redeveloped in the 1980s to become the Merry Hill Shopping
Centre.

 Brierley Hill was originally in Staffordshire, but is now part of the West Midlands metropolitan county since its creation in 1974.
</item 10>
=== END ITEM id=6b41aeb16e9f52c5 ===

=== ITEM id=7005d99b44f315c8 ===
Gold sentences and facts:
Sentence 0:  Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest commercial success is the erotic thriller "Chloe" (2009).
  Fact 0.0: Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter".
  Fact 0.1: "The Sweet Hereafter" was released in 1997.
  Fact 0.2: Egoyan received two Academy Award nominations for "The Sweet Hereafter".
  Fact 0.3: Egoyan's biggest commercial success is the erotic thriller "Chloe".
  Fact 0.4: "Chloe" was released in 2009.
Sentence 1: Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".
  Fact 1.0: Chloe is a 2009 erotic thriller film.
  Fact 1.1: Chloe was directed by Atom Egoyan.
  Fact 1.2: Chloe is a remake of the 2003 French film "Nathalie...".

Retrieved items:
<item 1>
Geoffrey Rush
Geoffrey Roy Rush {'1': ", '2': ", '3': ", '4': "} (born 6 July 1951) is an Australian actor and film producer.

 Rush is the youngest amongst the few people who have won the "Triple Crown of Acting": the Academy Award, the Primetime Emmy Award, and the Tony Award.

 He has won one Academy Award for acting (from four nominations), three British Academy Film Awards (from five nominations), two Golden Globe Awards and four
Screen Actors Guild Awards. Rush is the founding President of the Australian Academy of Cinema and Television Arts and was named the 2012 Australian of the
Year.

 He is also the first actor to win the Academy Award, BAFTA Award, Critics' Choice Movie Award, Golden Globe Award, and Screen Actors Guild Award for a single
performance in film for his performance in "Shine" (1996).
</item 1>

<item 2>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 2>

<item 3>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 3>

<item 4>
Graph query row:
person: Atom Egoyan
creative_work: Chloe
relationship: CREATED_BY
relationship_properties: {}
kind: film
published_on: 2009
</item 4>

<item 5>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 5>

<item 6>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 6>

<item 7>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 7>

<item 8>
Baltasar Kormákur
Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.

 He is best known for directing the films "101 Reykjavík", "Hafið", "A Little Trip to Heaven" (starring Julia Stiles and Forest Whitaker), a film based on the
book "Mýrin" "(Jar City)" by Arnaldur Indriðason, "Contraband", "2 Guns" (starring Mark Wahlberg and Denzel Washington) and "Everest".

 His father is the Spanish painter Baltasar Samper and actor Baltasar Breki Samper is his son.
</item 8>

<item 9>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 9>

<item 10>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 10>
=== END ITEM id=7005d99b44f315c8 ===
