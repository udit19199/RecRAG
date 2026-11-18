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

=== ITEM id=80e9981b33946d6b ===
Gold sentences and facts:
Sentence 0: Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale about a wily Irishman and his battle of wits with leprechauns.
  Fact 0.0: Darby O'Gill and the Little People is a feature film.
  Fact 0.1: Darby O'Gill and the Little People was made by Walt Disney Productions.
  Fact 0.2: Darby O'Gill and the Little People was released in 1959.
  Fact 0.3: Darby O'Gill and the Little People stars Albert Sharpe.
  Fact 0.4: Darby O'Gill and the Little People stars Janet Munro.
  Fact 0.5: Darby O'Gill and the Little People stars Sean Connery.
  Fact 0.6: Darby O'Gill and the Little People stars Jimmy O'Dea.
  Fact 0.7: Darby O'Gill and the Little People tells a tale about a wily Irishman.
  Fact 0.8: The Irishman in Darby O'Gill and the Little People has a battle of wits with leprechauns.
Sentence 1:  Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.
  Fact 1.0: The film named in this sentence was produced by Walt Disney Pictures.
  Fact 1.1: The film named in this sentence was produced by Aamir Khan Productions.
  Fact 1.2: The film named in this sentence was produced by UTV Motion Pictures.
  Fact 1.3: Aamir Khan stars as the father in the film named in this sentence.

Retrieved items:
<item 1>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 1>

<item 2>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 2>

<item 3>
Darby O'Gill
Darby O'Gill is a fictional Irishman who appears in the writings of British author Herminie Templeton Kavanagh, including her books "Darby O'Gill and the Good
People" (1903) and "Ashes of Old Wishes and Other Darby O'Gill Tales" (1926).
</item 3>

<item 4>
Darby O'Gill and the Little People
Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale
about a wily Irishman and his battle of wits with leprechauns.

 The film was directed by Robert Stevenson and its screenplay written by Lawrence Edward Watkin after the books of Herminie Templeton Kavanagh.
</item 4>

<item 5>
Albert Sharpe
Albert Sharpe (15 April 1885 – 13 February 1970) was an Irish stage and film actor.

 His most famous roles were those of Darby O'Gill in Disney's "Darby O'Gill and the Little People" and as Finian McLonergan in the Original Broadway production
of the musical "Finian's Rainbow".

 (the film version, made in 1968, had Fred Astaire in the role.)

 On screen he played Fiona's father Andrew in the MGM musical Brigadoon.

 He was also a member of the Abbey Players.

 His last ten years were spent in retirement.

 He died in 1970 in Belfast at the age of 84.
</item 5>

<item 6>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 6>

<item 7>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 7>

<item 8>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 8>

<item 9>
Walt Disney's Treasury of Classic Tales
Walt Disney's Treasury of Classic Tales is an American Sunday comic strip, which ran in newspapers from 13 July 1952 until 15 February 1987.

 Each story adapted a different Disney film, such as Darby O'Gill and the Little People, Peter Pan, or Davy Crockett.

 It was run in relatively few papers with 58 in 1957 and 55 in 1966 and was principally a vehicle for promoting new and re-released Disney films.
</item 9>

<item 10>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 10>
=== END ITEM id=80e9981b33946d6b ===

=== ITEM id=fc5fc9fe9e2359e1 ===
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
Bukovyna Airlines
Bukovyna Airlines, also known as Bukovyna Aviation Enterprise, is a charter airline based in Chernivtsi, Ukraine, operating chartered passenger flights out of
Chernivtsi International Airport.

 The company was founded in 1999.

 In 2013 Bukovyna was one of two Ukrainian airlines that had sanctions imposed against them by the US government.

 Bukovyna Airlines was hit with sanctions because it was leasing its US-built McDonnell Douglas MD-80 series aircraft to Iranian airlines Mahan Air and Iran
Air.

 The Iranian airlines were themselves under sanction by the US government.
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
relationship: None
event: 1993 Tehran mid-air collision
city: Tehran
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
=== END ITEM id=fc5fc9fe9e2359e1 ===

=== ITEM id=5a3b921248110fb2 ===
Gold sentences and facts:
Sentence 0: Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
  Fact 0.0: Martin Archer Flavin was born on November 2, 1883.
  Fact 0.1: Martin Archer Flavin died on December 27, 1967.
  Fact 0.2: Martin Archer Flavin was an American playwright.
  Fact 0.3: Martin Archer Flavin was an American novelist.
Sentence 1: Peter Bradford Benchley (May 8, 1940 – February 11, 2006) was an American author and screenwriter.
  Fact 1.0: Peter Bradford Benchley was born on May 8, 1940.
  Fact 1.1: Peter Bradford Benchley died on February 11, 2006.
  Fact 1.2: Peter Bradford Benchley was an American author.
  Fact 1.3: Peter Bradford Benchley was an American screenwriter.

Retrieved items:
<item 1>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 1>

<item 2>
Michael Breed
Michael S. Breed (born May 14, 1962, in Greenwich, Connecticut) is a professional golf instructor and television host.

 In 2003, he was selected as a Top 100 Instructor in America by "Golf Magazine", in 2011, he was voted one of the Top 50 Instructors in America by Golf Digest
(now 13th on this list, and #1 in the state of New York.)

, in 2012, Breed was chosen as the PGA's National Teacher of the Year.

 Prior to his time as a television host, Breed was the Head Golf Professional at Sunningdale Country Club for 12 years from 2001–2012, and the Head Golf
Professional at Birchwood Country Club, as well as the Assistant Professional at Deepdale Golf Club and Augusta National Golf Club.

 He has served on different boards for the Metropolitan PGA for over 10 years and has represented the PGA of America at the National Golf Day in Washington D.C.
since 2012.

 He is involved in charities such as The First Tee, Folds of Honor, Hope for the Warriors, Wounded Warriors Foundation and Salute Military Golf Association.
</item 2>

<item 3>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 3>

<item 4>
The entries profile actors, voice performers, puppeteers, composers, writers, and animation history. Ken Barrie voiced and sang for *Postman Pat*; Roger Kitter
played Captain Bertorelli in *’Allo ’Allo!* and scored a tennis-themed chart hit. Ralph Bakshi directed influential adult animations including *Fritz the Cat*,
*Heavy Traffic*, and *Fire and Ice*, plus live-action *Cool and the Crazy*. Other profiles include Disney composer Buddy Baker; Sc
</item 4>

<item 5>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 5>

<item 6>
Martin Flavin (politician)
Martin Flavin (1841– 30 December 1916) was an Irish nationalist politician, butter merchant and prominent businessman from Cork.

 He was a Member of Parliament (MP) from 1891 to 1892 .
</item 6>

<item 7>
Peter Benchley
Peter Bradford Benchley (May 8, 1940 – February 11, 2006) was an American author and screenwriter.

 He is known for the bestselling novel "Jaws" and co-wrote its subsequent film adaptation with Carl Gottlieb.

 Several more of his works were also adapted for cinema, including "The Deep", "The Island", "Beast", and "White Shark".
</item 7>

<item 8>
Martin Flavin
Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
</item 8>

<item 9>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 9>

<item 10>
Mark Benton
Mark Benton (born 16 November 1965) is an English actor, known for his roles as Eddie in "Early Doors", Howard in "Northern Lights" and Martin Pond in
"Barbara".

 Benton has also starred in BBC One drama, "Waterloo Road" as maths teacher, Daniel Chalk from 2011 to 2013 and again in 2014.
</item 10>
=== END ITEM id=5a3b921248110fb2 ===

=== ITEM id=9a270fdaf244dece ===
Gold sentences and facts:
Sentence 0:  It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style theme.
  Fact 0.0: The referenced piece plays during the opening sequence.
  Fact 0.1: Danny Elfman composed the piece in 1989.
  Fact 0.2: Danny Elfman composed the piece after Matt Groening approached him.
  Fact 0.3: Matt Groening is the series creator.
  Fact 0.4: Matt Groening requested a retro-style theme.
Sentence 1:  Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's Day Off".
  Fact 1.0: Clausen has scored or orchestrated music for more than 30 films and television shows.
  Fact 1.1: Clausen has scored or orchestrated music for 'Moonlighting'.
  Fact 1.2: Clausen has scored or orchestrated music for 'The Naked Gun'.
  Fact 1.3: Clausen has scored or orchestrated music for 'ALF'.
  Fact 1.4: Clausen has scored or orchestrated music for 'Ferris Bueller's Day Off'.

Retrieved items:
<item 1>
Alf Clausen
Alf Heiberg Clausen (born March 28, 1941) is an American film and television composer.

 He is best known for his work scoring many episodes of "The Simpsons", of which he had been the sole composer between 1990 and 2017.

 Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's
Day Off".
</item 1>

<item 2>
The Simpsons Theme
"The Simpsons" Theme", also referred to as "The Simpsons" Main Title Theme" in album releases, is the theme music of the animated television series "The
Simpsons".

 It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style
theme.

 The piece, which took 3 days, 2 hours, 48 minutes, and 19 seconds to create, has been noted by Elfman as the most popular of his career.

 The theme, as used for the opening sequence, was re-arranged during season 2, and the current arrangement by Alf Clausen was introduced at the beginning of the
third season.
</item 2>

<item 3>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 3>

<item 4>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 4>

<item 5>
Diwa de Leon
Diwa de Leon is a composer, arranger and musician based in Manila, Philippines.

 He makes music mainly for television shows, films, and stage productions.

 His most notable work is his arrangement of GMA Network's "Survivor Philippines" theme music.

 His songs included in "Emir".

 He won the Cinema One Originals Best Musical Score award for his work on the film "Kolorete" and Best Music Score for "Kamera Obskura" at the 8th Cinemalaya
Independent Film Festival.

 He is also a founding member of the world music group Makiling.

 His first mainstream film project is Mamarazzi produced by Regal Films.
</item 5>

<item 6>
Angelo Badalamenti
Angelo Badalamenti (born March 22, 1937) is an American composer, best known for his work scoring films for director David Lynch, notably "Blue Velvet", the
"Twin Peaks" saga (1990–1992, 2017), "The Straight Story" and "Mulholland Drive".

 Badalamenti received the 1990 Grammy Award for Best Pop Instrumental Performance for his ""Twin Peaks" Theme", and has received a Lifetime Achievement Award
from the World Soundtrack Awards and the Henry Mancini Award from the American Society of Composers, Authors and Publishers.
</item 6>

<item 7>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
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
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 9>

<item 10>
Disney's Adventures of the Gummi Bears
The show is well-remembered for its theme music, written by Michael and Patty Silversher and creation of "gummiberry juice" which was a type of magic potion,
granting abilities, which allowed them to bounce away from their hunters.

 The theme song was performed by Joseph Williams, son of composer John Williams and one-time lead singer of Toto.
</item 10>
=== END ITEM id=9a270fdaf244dece ===

=== ITEM id=e12beb1864063f95 ===
Gold sentences and facts:
Sentence 0:  The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
  Fact 0.0: The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck.
  Fact 0.1: Dave Brubeck was also one of the company's investors.
  Fact 0.2: The label is known more for its recordings of comedian Lenny Bruce.
  Fact 0.3: The label is known more for its recordings of jazz musician Vince Guaraldi.
  Fact 0.4: The label is known more for its recordings of the rock band Creedence Clearwater Revival.
Sentence 1: Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
  Fact 1.0: Vincent Anthony Guaraldi was born on July 17, 1928.
  Fact 1.1: Vincent Anthony Guaraldi died on February 6, 1976.
  Fact 1.2: Vincent Anthony Guaraldi was born Vincent Anthony Dellaglio.
  Fact 1.3: Vincent Anthony Guaraldi was an American jazz pianist.
  Fact 1.4: Vincent Anthony Guaraldi was noted for his innovative compositions and arrangements.
  Fact 1.5: Vincent Anthony Guaraldi was noted for composing music for animated television adaptations of the 'Peanuts' comic strip.
  Fact 1.6: Vincent Anthony Guaraldi performed on piano as a member of Cal Tjader's late 50s ensemble.
  Fact 1.7: Vincent Anthony Guaraldi had a solo career.
  Fact 1.8: Vincent Anthony Guaraldi's solo career included the radio hit 'Cast Your Fate to the Wind'.

Retrieved items:
<item 1>
Alan Young
Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television
host/personality who "TV Guide" called "The Charlie Chaplin of Television".

 He was best known for his role as naive Wilbur Post in the television comedy series "Mister Ed" (1961–1966).

 Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983)
and in various other films, TV series and video games until his death.

 During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy
Awards in 1951.

 He also appeared in a number of feature films, starting from 1946, including the 1960 film "The Time Machine" and from the 1980s gaining a new generation of
viewers appearing in numerous Walt Disney Productions films as both an actor and voice actor.
</item 1>

<item 2>
The Best of Creedence Clearwater Revival
The Best of Creedence Clearwater Revival is a compilation album by American rock band Creedence Clearwater Revival, released in 1977.

 The album features all the tracks of Chronicle which was released a year earlier in 1976, as well as "Good Golly Miss Molly", "Born on the Bayou", "Cotton
Fields", "Hello Mary Lou", "The Midnight Special", "Walk on the Water", and "Night Time Is the Right Time" which would appear on the 1986 release "Chronicle,
Vol.

 2", and "Bootleg" which does not appear on either volume.
</item 2>

<item 3>
Marc Blitzstein
Blitzstein also composed music for films, such as "Surf and Seaweed" (1931) and "The Spanish Earth" (1937), and he contributed two songs to the original 1960
production of Hellman's play "Toys in the Attic".
</item 3>

<item 4>
John Fogerty (album)
John Fogerty is the second solo studio album by former Creedence Clearwater Revival vocalist/guitarist John Fogerty, released in 1975.

 It was released by Asylum Records in the United States and Fantasy Records internationally.

 As with the Creedence Clearwater Revival records, the album consists of a mix of originals and cover songs.

 Although the album is eponymously titled, Fogerty himself refers to it as "Old Shep"; Shep was the name of his dog, who appears on the cover with him.
</item 4>

<item 5>
Fantasy Records
Fantasy Records is an American record company and label founded by brothers Max and Sol Weiss in 1949.

 The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known
more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
</item 5>

<item 6>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 6>

<item 7>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 7>

<item 8>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 8>

<item 9>
Jerry Beck
Peabody and Sherman", DreamWorks' "Madagascar", and "Bee Movie".

 Beck is also an entertainment industry consultant for TV and home entertainment productions and releases related to classic cartoons and operates the blog
"Cartoon Research."

 He appears frequently as a documentary subject and audio commentator on releases of A&E's "Cartoons Go to War" as well as DVD collections of "Looney Tunes",
"Popeye the Sailor", and "Woody Woodpecker" cartoons, on which he serves as consultant and curator.
</item 9>

<item 10>
Vince Guaraldi
Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions
and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of
Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
</item 10>
=== END ITEM id=e12beb1864063f95 ===

=== ITEM id=92815286dc42de12 ===
Gold sentences and facts:
Sentence 0: Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.
  Fact 0.0: Kim Dong-jun's Hangul name is 김동준.
  Fact 0.1: Kim Dong-jun's Hanja name is 金桐俊.
  Fact 0.2: Kim Dong-jun was born February 11, 1992.
  Fact 0.3: Kim Dong-jun is a South Korean singer.
  Fact 0.4: Kim Dong-jun is a South Korean actor.
Sentence 1: Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.
  Fact 1.0: Charlie Harper was born David Charles Perez.
  Fact 1.1: Charlie Harper was born on 25 May 1944.
  Fact 1.2: Charlie Harper was born in Hackney, London.
  Fact 1.3: Charlie Harper is a British singer.
  Fact 1.4: Charlie Harper is a British songwriter.

Retrieved items:
<item 1>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 1>

<item 2>
Two and a Half Men
Two and a Half Men is an American television sitcom that originally aired on CBS for twelve seasons from September 22, 2003 to February 19, 2015.

 Originally starring Charlie Sheen, Jon Cryer, and Angus T. Jones, the series was about a hedonistic jingle writer, Charlie Harper; his uptight brother Alan;
and Alan's troublesome son Jake.

 After Alan divorces, he moves with his son to share Charlie's beachfront Malibu house and complicates Charlie's freewheeling life.
</item 2>

<item 3>
Walden Schmidt
Walden Michael Thoreau Schmidt is a fictional character in the CBS sitcom "Two and a Half Men", portrayed by Ashton Kutcher.

 The character was introduced in the season nine premiere episode, "Nice to Meet You, Walden Schmidt" to replace the character of Charlie Harper after actor
Charlie Sheen was sacked from the series.

 Walden was a main character throughout the show's final four seasons (seasons 9-12).

 He is listed on Forbes' fictional list of the top 15 richest characters as number 11, with a wealth of approximately $1.3 billion.
</item 3>

<item 4>
Charlie Harper (singer)
Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.

 He is best known as the lead singer from the punk band U.K. Subs.
</item 4>

<item 5>
That Darn Priest
"That Darn Priest" is the season finale of the eighth season of the American sitcom "Two and a Half Men" and the final appearance of Charlie Sheen as Charlie
Harper.

 The episode is the 16th and final episode of the season, though it was meant to be the ninth to last episode, but the season was cut short due to Sheen's
problems related to drug and alcohol abuse.

 Three weeks after the episode's original airing, Sheen was officially fired from the show.

 CBS and Warner Bros. later announced that Ashton Kutcher would join the show's cast as Sheen's replacement.
</item 5>

<item 6>
Nangnangnangnang
"Nangnangnangnang" is the eleventh-season premiere of the comedy series "Two and a Half Men" and the first appearance of Amber Tamblyn as Charlie Harper's
estranged daughter, Jenny.

 The episode aired on September 26, 2013.
</item 6>

<item 7>
Collins &amp; Harlan
Collins & Harlan, the team of Arthur Collins and Byron G. Harlan formed a popular comic duet between 1903 and 1926.

 They sang ragtime standards as well as what were known as "Coon songs" – music sung by white performers in a black dialect.

 Their material also employed many other stereotypes of the time including Irishmen and farmers.

 Fellow recording artist Billy Murray nicknamed them "The Half-Ton Duo" as both men were rather overweight.

 Collins and Harlan produced many number one hits with recordings of minstrel songs such as "My Gal Irene", "I Know Dat I'll be Happy Til I Die", "Who Do You
Love?"

 and "Down Among the Sugarcane".

 Their song "That Funny Jas Band from Dixieland", recorded November 8, 1916, is among the first recorded uses of the word "jas" which eventually evolved to
"jass", and to the current spelling "jazz".
</item 7>

<item 8>
Kim Dong-jun (footballer)
Kim Dong-jun (Korean: 김동준 , born 19 December 1994) is a South Korean footballer who plays as a goalkeeper for Yonsei University and South Korea national team.
</item 8>

<item 9>
Nice to Meet You, Walden Schmidt
"Nice to Meet You, Walden Schmidt" is the ninth season premiere of the comedy series "Two and a Half Men" and the first appearance of Ashton Kutcher as Walden
Schmidt, "an internet billionaire with a broken heart".

 It is the 178th episode of the show, and the first episode without former lead Charlie Sheen as Charlie Harper.

 The episode was watched by 28.74 million people on its original air date, making it the most watched episode of "Two and a Half Men".
</item 9>

<item 10>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 10>
=== END ITEM id=92815286dc42de12 ===

=== ITEM id=11683c80b6c44537 ===
Gold sentences and facts:
Sentence 0:  After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.
  Fact 0.0: He spent some time in general practice.
  Fact 0.1: He became a Lecturer in animal behaviour at De Montfort University.
  Fact 0.2: He became a Senior Lecturer in animal behaviour at De Montfort University.
  Fact 0.3: He became a Principal Lecturer in animal behaviour at De Montfort University.
  Fact 0.4: He subsequently moved to the University of Lincoln.
  Fact 0.5: He has been Head of the Department of Biological Sciences at the University of Lincoln.
Sentence 1:  Individuals who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title of "Doctor" (often abbreviated "Dr") or, in non-English speaking countries, variants such as "Dr. phil."
  Fact 1.0: People who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title 'Doctor'.
  Fact 1.1: The title 'Doctor' is often abbreviated 'Dr'.
  Fact 1.2: In non-English speaking countries, people with a Doctor of Philosophy degree may use variants such as 'Dr. phil.'

Retrieved items:
<item 1>
Lincoln University College, Malaysia
Lincoln University College, Petaling Jaya, Malaysia, was founded in the year in 2002 as Lincoln College and in 2011 Lincoln College was upgraded to Lincoln
University College.

 Lincoln University College is a private institution that provides higher education, approved by the Malaysian Qualifications Agency (MQA, National
Accreditation Board) and the Ministry of Higher Education.

 Situated in the cosmopolitan town of Petaling Jaya, the university college is close to the capital city, Kuala Lumpur.
</item 1>

<item 2>
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
</item 2>

<item 3>
William McGinnis
William "Bill" McGinnis, Ph.D. is a molecular biologist and professor of biology at the University of California San Diego.

 At UC San Diego he has also served as the Chairman of the Department of Biology from July 1998 - June 1999, as Associate Dean of the Division of Natural
Sciences from July 1, 1999 - June 2000, and as Interim Dean of the newly established Division of Biological Sciences from July 1, 2000 - February 1, 2001.

 Dr. McGinnis was appointed Dean of the Divisional Biological Sciences on July 1, 2013
</item 3>

<item 4>
David M. Sever
David M. Sever is an American herpetologist, histologist, anatomist and reproductive biologist.

 He has been a professor and department head in the Department of Biological Sciences at Southeastern Louisiana University since 2004, and held the Kenneth
Dyson Endowed Professorship in Biological Sciences from 2012 to 2015.

 He is well known for over 30 years of research on the secondary sexual characteristics of salamanders and more generally on comparative histoanatomy of the
urogenital systems of vertebrates.

 and was recognized as the 2013 Distinguished Herpetologist of the Year by the Herpetologists' League.
</item 4>

<item 5>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 5>

<item 6>
Daniel Mills (biologist)
Daniel Simon Mills, FRCVS (born 21 Aug 1966) is an English biologist and the UK's first Professor of Veterinary Behavioural Medicine based at the University of
Lincoln, United Kingdom.

 He attended St Edmund's College, Ware before studying at the University of Bristol where he received his BVSc degree and completed his Ph.D. degree in animal
behaviour from De Montfort University.

 After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and
subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.

 His research has focused on the assessment of emotion in companion animals and the use of semiochemicals to manage their problem behaviour (pheromonatherapy).
</item 6>

<item 7>
Richard F. Wilson
Richard F. Wilson was selected to be Illinois Wesleyan University's 18th President on April 6, 2004 and took office on July 1, 2004.

 He was inaugurated on April 9, 2005.

 He had spent the previous 26 years at the University of Illinois at Urbana-Champaign, leaving as associate chancellor for development and vice president of the
University of Illinois Foundation.

 Mr. Wilson is a Ph.D. graduate of the University of Michigan.

 Wilson announced his intended retirement from Illinois Wesleyan, to take place around June 2015 or whenever a successor is named and transitioned.
</item 7>

<item 8>
Animal Demography Unit
The Animal Demography Unit (ADU) is a formally recognized research unit of the University of Cape Town (UCT) located within the Department of Biological
Sciences of UCT.

 (The Department of Biological Sciences was formed from the merger of the Department of Botany and the Department of Zoology at the start of the 2013 academic
year).

 The Animal Demography Unit, popularly known as the ADU, was responsible for the management of the First and Second Southern African Bird Atlas Projects SABAP1
and SABAP2.

 The unit has submitted over eight million georeferenced biodiversity records to GBIF.
</item 8>

<item 9>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 9>

<item 10>
Wale Adebanwi
Professor Wale Adebanwi, (born 1969), is a Nigerian - born first Black African Rhodes Professor at Oxford University
</item 10>
=== END ITEM id=11683c80b6c44537 ===

=== ITEM id=163e6194e1a9b915 ===
Gold sentences and facts:
Sentence 0: Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.
  Fact 0.0: Jelena Janković was born on 28 February 1985.
  Fact 0.1: Jelena Janković's name is written in Serbian Cyrillic as Јелена Јанковић.
  Fact 0.2: Jelena Janković is a Serbian professional tennis player.
Sentence 1: Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian politician and retired professional tennis player.
  Fact 1.0: Marat Mubinovich Safin's name is written in Russian as Мара́т Муби́нович Са́фин.
  Fact 1.1: Marat Mubinovich Safin's name is written in Tatar as Марат Мөбин улы Сафин.
  Fact 1.2: Marat Mubinovich Safin's Tatar name is romanized as Marat Möbin uğlı Safin.
  Fact 1.3: Marat Mubinovich Safin was born on 27 January 1980.
  Fact 1.4: Marat Mubinovich Safin is a Russian politician.
  Fact 1.5: Marat Mubinovich Safin is a retired professional tennis player.

Retrieved items:
<item 1>
Jelena Janković career statistics
This is a list of the main career statistics of Serbian professional tennis player and former World No. 1, Jelena Janković.

 To date, Janković has won 15 WTA singles titles including four WTA Tier I singles titles, one WTA Premier Mandatory singles title and one WTA Premier 5 singles
title.

 Janković was also the runner-up in singles at the 2008 US Open, a semi-finalist at the year-ending WTA Tour Championships in 2008, 2009 and 2013 and a
quarterfinalist at the 2008 Beijing Olympics.
</item 1>

<item 2>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 2>

<item 3>
Dinara Safina career statistics
This is a list of the main career statistics of retired, Russian professional tennis player, Dinara Safina.

 Throughout her career, Safina won twelve WTA singles titles including three Tier I singles titles at the 2008 Qatar Telecom German Open, Rogers Cup and Toray
Pan Pacific Open respectively; one Premier Mandatory singles title at the 2009 Mutua Madrileña Madrid Open and one Premier 5 singles title at the 2009
Internazionali BNL d'Italia.

 She was also the runner-up at the 2008 French Open and the 2009 Australian Open and French Open as well as a silver medalist in singles at the 2008 Beijing
Olympics.
</item 3>

<item 4>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 4>

<item 5>
Ana Ivanovic career statistics
This is a list of the main career statistics of Serbian professional tennis player, Ana Ivanovic.

 Ivanovic won fifteen WTA singles titles including one grand slam singles title at the 2008 French Open and three WTA Tier I singles titles.

 She was also the runner-up at the 2007 French Open and 2008 Australian Open and a semi-finalist at the 2007 Wimbledon Championships and 2007 WTA Tour
Championships.

 On June 9, 2008 Ivanovic became the world No. 1 for the first time in her career.
</item 5>

<item 6>
2008 Wimbledon Championships – Mixed Doubles
Jamie Murray and Jelena Janković were the defending champions, but Janković chose not to participate, and only Murray competed that year.
</item 6>

<item 7>
Jelena Janković
Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.

 Janković is a former World No. 1 in singles, a ranking achieved preceding her finals appearance at the 2008 US Open.

 Janković's career highlights include winning the 2007 Wimbledon Mixed Doubles title with Jamie Murray, the 2010 Indian Wells Masters, the Internazionali BNL
d'Italia twice, in 2007 and 2008, and the aforementioned appearance in the finals of the 2008 US Open.

 She is coached by her brother Marko.
</item 7>

<item 8>
Ivanovic–Janković rivalry
Serbian tennis players Ana Ivanovic and Jelena Janković are both former world No. 1 and have met 12 times so far.

 Their overall head-to-head series is 9–3, in Ivanovic's favor.

 Janković is two years older than Ivanovic, so their paths connect them to play each other.

 They are the first female players to achieve major successes for independent Serbia.
</item 8>

<item 9>
Graph query row:
person: Marat Safin
occupation: ['politician', 'retired professional tennis player']
nationality: Russian
birth_date: 27 January 1980
</item 9>

<item 10>
Graph query row:
person: Jelena Janković
occupation: professional tennis player
nationality: Serbian
birth_date: 28 February 1985
</item 10>
=== END ITEM id=163e6194e1a9b915 ===

=== ITEM id=2cf746563b60cd2a ===
Gold sentences and facts:
Sentence 0:  In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
  Fact 0.0: In 2003, a person referred to as "he" received an award for his actions during the Battle of Qala-i-Jangi.
  Fact 0.1: The Battle of Qala-i-Jangi took place in late November to early December 2001.
Sentence 1: The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November 25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
  Fact 1.0: The Battle of Qala-i-Jangi is also incorrectly referred to as the "Battle of Mazar-i-Sharif".
  Fact 1.1: The Battle of Qala-i-Jangi was a prisoner-of-war camp uprising.
  Fact 1.2: The Battle of Qala-i-Jangi took place between November 25 and December 1, 2001.
  Fact 1.3: The Battle of Qala-i-Jangi took place in northern Afghanistan.
  Fact 1.4: The Battle of Qala-i-Jangi followed the armed intervention by United States-led coalition forces.
  Fact 1.5: The United States-led coalition forces intervened with armed forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan.
  Fact 1.6: The coalition forces had accused the Taliban's Islamic Emirate of Afghanistan of harboring al-Qaeda operatives.

Retrieved items:
<item 1>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 1>

<item 2>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 2>

<item 3>
National Intelligence Reform Medal
The National Intelligence Reform Medal is an award of the United States Intelligence Community's National Intelligence Awards Program that recognizes
extraordinary service in implementing the Intelligence Reform and Terrorism Prevention Act.

 The medal ranks below the National Intelligence Superior Service Medal, but above the National Intelligence Exceptional Achievement Medal.

 Established 23 May 2007 with the creation of the National Intelligence Awards Program, it was retired in November 2010.
</item 3>

<item 4>
Battle of Qala-i-Jangi
The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November
25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's
Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
</item 4>

<item 5>
Sayeed Salahudeen
Syed Mohammed Yusuf Shah, popularly known as Syed Salahudeen, is the head of Hizb-ul-Mujahideen, a terrorist group operating in Kashmir, and head of an alliance
of anti-India militant groups, the United Jihad Council, that works to annex the Indian state of Jammu and Kashmir to Pakistan .

 Salahuddin vowed to block any peaceful resolution to the Kashmir conflict, threatened to train more Kashmiri suicide bombers, and vowed to turn the Kashmir
valley “into a graveyard for Indian forces.”

 He is listed on the NIA Most Wanted list.

 On 26 June 2017, he was named as a Specially Designated Global Terrorist by the US Department Of State.

 After being designated as a Global terrorist, he addressed the Pakistani Media at Muzaffarabad’s Centre Press Club, claiming that "The declaration was a joint
move by the US, Israel, and India to express their animosity towards Pakistan."
</item 5>

<item 6>
Enemy Combatant (book)
Enemy Combatant is a memoir by British Muslim, Moazzam Begg, co-written by Victoria Brittain, former Associate Foreign Editor for "The Guardian", about Begg's
detention by the government of the United States of America in Bagram Detention Facility and at Camp Echo, Guantanamo Bay and his life prior to that detention.

 It was published in Britain as "Enemy Combatant: A British Muslim's Journey To Guantanamo and Back" (ISBN  ), and in the US as "Enemy Combatant: My
Imprisonment at Guantanamo, Bagram, and Kandahar" (ISBN  ).

 In the US, the foreword was written by David Ignatius of The Washington Post.
</item 6>

<item 7>
The Good Terrorist
The Good Terrorist is a 1985 political novel written by the British novelist Doris Lessing.

 It was first published in September that year by Jonathan Cape in the United Kingdom and by Alfred A. Knopf in the United States.

 The book's protagonist is the naïve drifter Alice, who squats with a group of radicals in London and is drawn into their terrorist activities.
</item 7>

<item 8>
Soldiers of Egypt
Soldiers of Egypt (Ajnad Misr in Arabic) is an active Salafist Islamist militant group that has been operating near Cairo, Egypt.

 The groups was founded by Humam Muhammed in 2013, after he split away from the Ansar Bait al-Maqdis militant group.

 The group claims that its attacks are "retribution" for the August 2013 Rabaa Massacre; notably, the group targets only security forces.

 It has warned civilians of the presence of bombs that it has placed.

 The Cairo Court for Urgent Matters declared the group a terrorist group on 22 May 2014.

 It has been a Proscribed Organisation in the United Kingdom under the Terrorism Act 2000 since November 2014.

 The United States Department of State designated it a terrorist organization on 18 December 2014.
</item 8>

<item 9>
Combat Terrorist Organization
Combat Terrorist Organization or Autonomous Combat Terrorist Organization (rus.

 Боевая Террористическая Организация) was a Russian radical Neo-Nazi terrorist group based in Saint Petersburg from 9 August 2003 to 2006.

 It was formed by two members of the "Mad Crowd" skinhead organization, Dmitriy Borovikov and Voevodin Alexey.
</item 9>

<item 10>
Iraultza
Iraultza (Basque for Revolution) was a small Basque militant armed group of leftist tendency.

 Seven of the group would die preparing explosive devices.

 On June 27, 1986, a bomb claimed by Iraultza mangled a young worker called José Miguel Moros Peña in Portugalete.

 The victim would die in Cruces Hospital (Barakaldo) on August 13, 1986.

 The group is also known for being described by the United States Government as "probably [having] committed more bombings against American business interests
than any other European terrorist group".

 Soon after, the group, which had been active for about a decade with small attacks of explosives that never aimed to cause any personal injuries, announced its
dissolution.

 When the group was operational, it is estimated that it had approximately 20 members.
</item 10>
=== END ITEM id=2cf746563b60cd2a ===

=== ITEM id=1a49a50d45c03927 ===
Gold sentences and facts:
Sentence 0:  He has been involved in numerous television projects as director, writer, producer and animator.
  Fact 0.0: The person described has been involved in numerous television projects.
  Fact 0.1: The person described has worked on television projects as a director.
  Fact 0.2: The person described has worked on television projects as a writer.
  Fact 0.3: The person described has worked on television projects as a producer.
  Fact 0.4: The person described has worked on television projects as an animator.
Sentence 1: Béla Gaál (2 January 1893 – 18 February 1945) was a Hungarian film director.
  Fact 1.0: Bela Gaal was born on 2 January 1893.
  Fact 1.1: Bela Gaal died on 18 February 1945.
  Fact 1.2: Bela Gaal was a Hungarian film director.

Retrieved items:
<item 1>
Heavy Traffic
Heavy Traffic is a 1973 American adult animated comedy-drama film written and directed by Ralph Bakshi.

 The film, which begins, ends, and occasionally combines with live-action, explores the often surreal fantasies of a young New York cartoonist named Michael
Corleone, using pinball imagery as a metaphor for inner-city life.

 "Heavy Traffic" was Bakshi and producer Steve Krantz's follow-up to the film "Fritz the Cat".

 Though producer Krantz made varied attempts to produce an R-rated film, "Heavy Traffic" was given an X rating by the MPAA.

 The film received positive reviews and is widely considered to be Bakshi's biggest critical success.
</item 1>

<item 2>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 2>

<item 3>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 3>

<item 4>
Tales of Budapest
Tales Of Budapest (Hungarian: "Pesti mese" ) is a 1937 Hungarian comedy film directed by Béla Gaál.
</item 4>

<item 5>
Radha Bharadwaj
Radha Bharadwaj is an Indian filmmaker, film producer and screenwriter.

 She moved to the United States in her late teens to study film.

 Bharadwaj's screenwriting and directing feature debut is "Closet Land".

 The surreal psychological drama was released by Universal Pictures in 1991, making Radha Bharadwaj the first director of Indian descent to have a film released
by a major Hollywood studio.

 "Closet Land" stars Alan Rickman and Madeleine Stowe.

 Ron Howard and Brian Grazer produced the feature.

 The screenplay for "Closet Land" won the Nicholl Screenwriting Fellowship sponsored by the Academy of Motion Picture Arts and Sciences, among others.
</item 5>

<item 6>
Béla Gaál
Béla Gaál (2 January 1893 – 18 February 1945) was a Hungarian film director.

 His 1930 film "Csak egy kislány van a világon" was the first sound film to be made in Hungary.
</item 6>

<item 7>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 7>

<item 8>
Graph query row:
person1: Ralph Bakshi
occupation1: writer and director
person2: Béla Gaál
occupation2: director
shared_role: None
</item 8>

<item 9>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 9>

<item 10>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 10>
=== END ITEM id=1a49a50d45c03927 ===

=== ITEM id=e69928e01348b2cf ===
Gold sentences and facts:
Sentence 0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.
  Fact 0.0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen.
  Fact 0.1: Not Richard, But Dick was released in 1993.
  Fact 0.2: Not Richard, But Dick was released by Hollywood Records.
Sentence 1: The Dead Milkmen are an American punk rock band formed in 1983 in Philadelphia, Pennsylvania.
  Fact 1.0: The Dead Milkmen are an American punk rock band.
  Fact 1.1: The Dead Milkmen were formed in 1983.
  Fact 1.2: The Dead Milkmen were formed in Philadelphia, Pennsylvania.

Retrieved items:
<item 1>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 1>

<item 2>
Seasoned Veteran
Seasoned Veteran is the third studio album by American rapper Richie Rich.

 It was released November 5, 1996 on Def Jam Recordings, and to date is the only major label release for the artist.

 The album was produced by Ali Malik, DJ Daryl, Doug Rasheed, Jermaine Dupri, Lev Berlak, Mike Mosley, Richie Rich and Rick Rock.

 It peaked at number 11 on the "Billboard" Top R&B/Hip-Hop Albums and at number 35 on the "Billboard" 200.

 Two of the singles released, "Let's Ride" and "Do G's Get To Go To Heaven?

", both appeared on multiple "Billboard" singles charts.

 The album features guest performances by 2Pac, E-40, Luniz, D'wayne Wiggins, T-Boz, and Rame Royal of Rhythm & Green.
</item 2>

<item 3>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 3>

<item 4>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 4>

<item 5>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 5>

<item 6>
Electric Guitars
Electric Guitars were an English band formed early in 1980 by Neil Davenport (vocals, lyrics) and Richard Hall (bass, vocals) who were both studying English at
Bristol University.

 The band soon increased to a five-man line-up, with Andy Saunders (guitar, vocals), Matt Salt (drums) and Dick Truscott (keyboards), they also later added two
backing singers: Sara and Wendy Patridge.

 Their first single "Health" / "Continental Shelf" was released on local label Fried Egg Records in 1980.
</item 6>

<item 7>
Dick Delicious and The Tasty Testicles
Dick Delicious and The Tasty Testicles are a comedy / metal band from Atlanta, Georgia.

 The band formed in 1992 and released 3 albums between 1995 and 2003.

 In 2003 the Dick Delicious received the "Howard Stern Award for Musical Excellence" and broke up shortly afterward.
</item 7>

<item 8>
Not Richard, But Dick
Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.

 After being out of print for years, Hollywood Records decided to release the album for digital download as of April 2, 2013.
</item 8>

<item 9>
Graph query row:
work: Not Richard, But Dick
relationship: CREATED_BY
organization: The Dead Milkmen
city: Philadelphia
</item 9>

<item 10>
The Idols with Sid Vicious
The Idols with Sid Vicious is a concert album of former Sex Pistols bassist Sid Vicious performing with The Idols; though recorded in September 1978, it wasn't
released until 1993.

 Steve Dior provided a recording of this performance to the New Rose record label.

 It was originally released by the Fan Club division of New Rose.

 It was also released in Japan by Teichiku Records, under license from New Rose.
</item 10>
=== END ITEM id=e69928e01348b2cf ===
