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

=== ITEM id=c22c65f22e7a531b ===
Gold sentences and facts:
Sentence 0: Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.
  Fact 0.0: Summer Magic is a 1963 Walt Disney Productions film.
  Fact 0.1: Summer Magic stars Hayley Mills.
  Fact 0.2: Summer Magic stars Burl Ives.
  Fact 0.3: Summer Magic stars Dorothy McGuire.
  Fact 0.4: Summer Magic is a story about a Boston widow and her children.
  Fact 0.5: In Summer Magic, the Boston widow and her children take up residence in a small town in Maine.
Sentence 1: Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.
  Fact 1.0: Around the World in 80 Days is a 2004 American film.
  Fact 1.1: Around the World in 80 Days is an action-adventure comedy family film.
  Fact 1.2: Around the World in 80 Days is based on Jules Verne's novel of the same name.

Retrieved items:
<item 1>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 1>

<item 2>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 2>

<item 3>
Around the World (1956 song)
"Around the World" was the theme tune from the 1956 movie "Around the World in 80 Days" In the film, only an instrumental version of the song appeared, although
the vocal version has become by far the better known one.

 The song was written by Harold Adamson and Victor Young; Young died in 1956, several weeks after the film's release and he received the Academy Award for Best
Music, Scoring of a Dramatic or Comedy Picture posthumously.

 Young's orchestral version was a #13 hit on the Billboard charts in 1957.

 The recording by Bing Crosby was the B-side of the Victor Young version in 1957, on Festival SP45-1274 in Australia, and was a joint charting success.
</item 3>

<item 4>
Around the World in 80 Days (1956 film)
Around the World in 80 Days (sometimes spelled as Around the World in Eighty Days) is a 1956 American epic adventure-comedy film starring Cantinflas and David
Niven , produced by the Michael Todd Company and released by United Artists.
</item 4>

<item 5>
Summer Holiday (1963 film)
Summer Holiday is a British CinemaScope and Technicolor musical film featuring singer Cliff Richard.

 The film was directed by Peter Yates (his debut), produced by Kenneth Harper.

 The original screenplay was written by Peter Myers and Ronald Cass (who also wrote most of the song numbers and lyrics).

 The cast includes Lauri Peters, Melvyn Hayes, Teddy Green, Jeremy Bulloch, Una Stubbs, Pamela Hart, Jacqueline Daryl, Lionel Murton, Madge Ryan, David Kossoff,
Nicholas Phipps, Ron Moody and The Shadows.

 Herbert Ross choreographed the musical numbers.
</item 5>

<item 6>
The Good Old Days (film)
The Good Old Days is a 1939 British comedy film directed by Roy William Neill.

 Written by Austin Melford and John Dighton based on a story by Ralph Smart, it stars Max Miller, Hal Walters and Kathleen Gibson.

 The film tells the story of group of entertainers struggling to obtain permission to perform at a tavern in 1840.
</item 6>

<item 7>
Around the World in 80 Days (1988 film)
Around the World in 80 Days is an Australian 48-minute direct-to-video animated film from Burbank Films Australia.

 It was originally released in 1988.

 The film is based on Jules Verne's classic French novel, "Around the World in 80 Days", first published in 1873, and was adapted by Leonard Lee.

 It was produced by Roz Phillips and featured original music by Simon Walker.

 The film imitated BRB Internacional's Spanish 1981 series, "La vuelta al mundo de Willy Fog", in its use of anthropomorphic animals in the human roles.

 The copyright in this film is now owned by Pulse Distribution and Entertainment and administered by digital rights management firm NuTech Digital.
</item 7>

<item 8>
Around the World in Eighty Days
Around the World in Eighty Days (French: "Le tour du monde en quatre-vingts jours" ) is a classic adventure novel by the French writer Jules Verne, published in
1873.

 In the story, Phileas Fogg of London and his newly employed French valet Passepartout attempt to circumnavigate the world in 80 days on a £20,000 wager (the
approximate equivalent of £2 million in 2016) set by his friends at the Reform Club.

 It is one of Verne's most acclaimed works .
</item 8>

<item 9>
Summer Magic
Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children
taking up residence in a small town in Maine.

 The film was based on the novel "Mother Carey's Chickens" by Kate Douglas Wiggin and was directed by James Neilson.

 This was the fourth of six film Mills did for Disney, and the young actress received a Golden Globe nomination for her work here.
</item 9>

<item 10>
Around the World in 80 Days (2004 film)
Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.

 It stars Jackie Chan, Steve Coogan and Cécile de France.

 The film is set in 19th-century Britain and centers on Phileas Fogg (Steve Coogan), here reimagined as an eccentric inventor, and his efforts to circumnavigate
the globe in 80 days.

 During the trip, he is accompanied by his Chinese valet, Passepartout (Jackie Chan).

 For comedic reasons, the film intentionally deviated wildly from the novel and included a number of anachronistic elements.

 With production costs of about $110 million and estimated marketing costs of $30 million, it earned $24 million at the U.S. box office and $72 million
worldwide, making it a box office flop.

 It was Arnold Schwarzenegger's last film before he took a hiatus from acting to become Governor of California until 2010's "The Expendables".
</item 10>
=== END ITEM id=c22c65f22e7a531b ===

=== ITEM id=cef5a2853d0f6616 ===
Gold sentences and facts:
Sentence 0:  It is found in Yemen and India.
  Fact 0.0: The subject of this sentence is found in Yemen.
  Fact 0.1: The subject of this sentence is found in India.
Sentence 1:  Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.
  Fact 1.0: Yemen's capital has been temporarily relocated to the port city of Aden.
  Fact 1.1: Aden is on the southern coast.
  Fact 1.2: The relocation is stated to be because of the content of the previous sentence.

Retrieved items:
<item 1>
Aulacodes traversalis
Aulacodes traversalis is a moth in the Crambidae family.

 It was described by Harrison Gray Dyar Jr. in 1914.

 It is found in Panama.
</item 1>

<item 2>
Amka
Amka (Hebrew: עַמְקָה‎ ), also known in Arabic as Amqa (Arabic: عمقا‎ ‎ ), is a moshav in the Matte Asher Regional Council of Israel's Northern District, near
Acre.

 The location of the moshav roughly corresponds the former Arab village, depopulated during the 1948 Arab–Israeli War.

 Yemenite Jews, who arrived from the southern Arab country of Yemen, founded the village's successor Amka in 1949.

 In 2016 its population was 706 .
</item 2>

<item 3>
Al Khor (city)
Al Khor is a coastal city in northern Qatar, located 50 kilometres north of the capital, Doha.

 It is the capital city of the municipality of Al Khor.

 The name of the city means creek in Arabic as the town is located on a creek.

 Al Khor is home to many employees of the oil industry due to its proximity to Qatar's northern oil and natural gas fields, and due to its proximity to the Ras
Laffan Industrial City.
</item 3>

<item 4>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 4>

<item 5>
Abd Allah Siraj
Siraj served as acting Prime Minister in lieu of Emir Ali until 1918.

 After Husayn abdicated the throne in 1924, Siraj held the office of Prime Minister during most of Ali's short reign, which ended with the Kingdom's surrender
to the Saudi Sultanate of Nejd in 1925.

 He then migrated to the Jordan, where under Emir Abd Allah he served as Prime Minister from 1931 to 1933 while simultaneously holding the portfolios of Finance
and the Interior Ministry, as well as the office of Chief Justice.
</item 5>

<item 6>
Agrotis orthogonia
The Pale Western Cutworm ("Agrotis orthogonia") is a moth of the Noctuidae family.

 It is found in North America, more specifically dry, semi-desert areas of western North America from southern Canada to California, ranging eastward nearly to
the eastern edge of the Great Plains.
</item 6>

<item 7>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 7>

<item 8>
Ta'if Regional Airport
Ta'if Regional Airport (IATA: TIF, ICAO: OETF) is an airport in Ta'if, Saudi Arabia.

 Despite its name, it also offers many international flights as well.

 The airport is located 30 km to the east of Taif and 70 km from Mecca.

 The airport is considered important in Saudi aviation history as it witnessed the first landing of Ibn Saud's plane, founder of the Kingdom.

 It was converted to a regional airport in 2009 when GACA allowed international airlines to operate at the airport as the city's population was increasing and
to reduce pressure on the three main airports at the time.

 Despite being named as a "Regional" airport, the airport actually has international flights destinations in half a dozen countries outside of Saudi Arabia.
</item 8>

<item 9>
Aulacodes templalis
Aulacodes templalis is a moth in the Crambidae family.

 It was described by Schaus in 1906.

 It is found in Brazil.
</item 9>

<item 10>
Yemen
Yemen ( ; Arabic: اليَمَن‎ ‎ "al-Yaman "), officially known as the Republic of Yemen (الجمهورية اليمنية "al-Jumhūrīyah al-Yamanīyah "), is an Arab country in
Western Asia at the southern end of the Arabian Peninsula.

 Yemen is the second-largest country in the peninsula, occupying 527,970 km (203,850 sq mi).

 The coastline stretches for about 2,000 km (1,200 mi).

 It is bordered by Saudi Arabia to the north, the Red Sea to the west, the Gulf of Aden and Arabian Sea to the south, and Oman to the east-northeast.

 Although Yemen's constitutionally stated capital is the city of Sana'a, the city has been under rebel control since February 2015.

 Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.

 Yemen's territory includes more than 200 islands; the largest of these is Socotra.
</item 10>
=== END ITEM id=cef5a2853d0f6616 ===

=== ITEM id=a6f0450c98e7c416 ===
Gold sentences and facts:
Sentence 0:  Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.
  Fact 0.0: Most figures are shaped with the body of Mickey Mouse.
  Fact 0.1: Most figures have different themed markings.
  Fact 0.2: Most figures have different themed colors.
  Fact 0.3: Most figures have different themed patterns.
Sentence 1:  He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.
  Fact 1.0: Mickey Mouse was created by Walt Disney.
  Fact 1.1: Mickey Mouse was created by Ub Iwerks.
  Fact 1.2: Mickey Mouse was created at the Walt Disney Studios.
  Fact 1.3: Mickey Mouse was created in 1928.

Retrieved items:
<item 1>
Vinylmation
Vinylmation is a brand of 1.5", 3", and 9" vinyl collectibles sold at Disney theme parks, select Disney Stores, and the online Disney Store.

 The name "Vinylmation" is a combination of the word "Animation" and "Vinyl".

 Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.

 This means that all figures have round mouse ears, whether or not the painted character has them.

 This is accommodated by painting the ears with backgrounds - stars, stripes, rainbows - to match the painted character.

 The series exception is the Park Starz series which does not resemble Mickey at all, and more closely resembles iconic figures from the parks.

 Vinylmation was first introduced in July 2008, and the first figures were introduced in November 2008.
</item 1>

<item 2>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 2>

<item 3>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 3>

<item 4>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 4>

<item 5>
Mickey Mouse
Mickey Mouse is a funny animal cartoon character and the official mascot of The Walt Disney Company.

 He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.

 An anthropomorphic mouse who typically wears red shorts, large yellow shoes, and white gloves, Mickey has become one of the world's most recognizable
characters.
</item 5>

<item 6>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 6>

<item 7>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 7>

<item 8>
Felix the Cat
Felix the Cat is a funny-animal cartoon character created in the silent film era.

 The anthropomorphic black cat with his black body, white eyes, and giant grin, coupled with the surrealism of the situations in which his cartoons place him,
combine to make Felix one of the most recognized cartoon characters in film history.

 Felix was the first character from animation to attain a level of popularity sufficient to draw movie audiences.
</item 8>

<item 9>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 9>

<item 10>
Donald Duck
Donald Duck is a cartoon character created in 1934 at Walt Disney Productions.

 Donald is an anthropomorphic white duck with a yellow-orange bill, legs, and feet.

 He typically wears a sailor shirt and cap with a bow tie.

 Donald is most famous for his semi-intelligible speech and his mischievous and temperamental personality.

 Along with his friend Mickey Mouse, Donald is one of the most popular Disney characters and was included in TV Guide's list of the 50 greatest cartoon
characters of all time in 2002.

 He has appeared in more films than any other Disney character, and is the most published comic book character in the world outside of the superhero genre.
</item 10>
=== END ITEM id=a6f0450c98e7c416 ===

=== ITEM id=1aeb3089a4dde0c1 ===
Gold sentences and facts:
Sentence 0: Dana Robins Ivey (born August 12, 1941) is an American actress.
  Fact 0.0: Dana Robins Ivey was born on August 12, 1941.
  Fact 0.1: Dana Robins Ivey is an American actress.
Sentence 1: Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.
  Fact 1.0: Two Weeks Notice is a 2002 film.
  Fact 1.1: Two Weeks Notice is an Australian-American film.
  Fact 1.2: Two Weeks Notice is a romantic comedy film.
  Fact 1.3: Two Weeks Notice stars Hugh Grant.
  Fact 1.4: Two Weeks Notice stars Sandra Bullock.

Retrieved items:
<item 1>
Canadian actress Rachelle Lefevre starred in *Big Wolf on Campus*, played Victoria in the first two *Twilight* films, and appeared in *Off the Map*, *A Gifted
Man*, and *Under the Dome*. American actress Gloria Garayua debuted in *Fun with Dick and Jane* and had roles in *Grey’s Anatomy*, *Cougar Town*, and *How to
Get Away with Murder*. Zooey Deschanel’s films include *Almost Famous*, *All
</item 1>

<item 2>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 2>

<item 3>
Josann McGibbon
Josann McGibbon is an American screenwriter working in partnership with Sara Parriott.

 The team's first major success as a screenwriter was the early Brad Pitt film, "The Favor".

 Their biggest hits since then include "Three Men and a Little Lady" and "Runaway Bride".

 In 2007, McGibbon and Parriott co-wrote and produced the hit Debra Messing miniseries, "The Starter Wife".

 "The Starter Wife" received 10 Emmy nominations in 2007, including for best screenwriting, and won one Emmy Award.

 It was also nominated for Golden Globe and Writers Guild awards, and was then produced as a series, also on USA Network.

 McGibbon and Parriott wrote and co-produced the Disney Channel movie, "Descendants" which was directed by Kenny Ortega and premiered in July, 2015.

 On February, 2013, it won the Writers Guild of America Award in television for Outstanding Children's Long Form.

 They then wrote and executive-produced "Descendants 2", also directed by Kenny Ortega, which was simulcast on the Disney Channel, ABC, and the other cable
channels owned by Disney-ABC in July, 2017.
</item 3>

<item 4>
Two Weeks (The Office)
"Two Weeks" is the twenty-first episode of the fifth season of the television series "The Office", and the 93rd overall episode of the series.

 It originally aired on NBC in the United States on March 26, 2009.

 In this episode, Michael, who has given his two weeks' notice to Dunder Mifflin, tries to convince others in the office to quit and join him in starting a new
paper company.

 Meanwhile, Pam spends her day trying to put together the new photocopier and becomes frustrated with her job.
</item 4>

<item 5>
Dana (surname)
Dana ( or ) as a surname may have several origins.

 In England, it came from "dann", the valley of a meadow, and it may mean the dweller of that valley.

 In Continental Europe, it probably came from "Dane" (or Danish, from Denmark).

 This surname is related to Danese in Italy and it can be found mostly in the Piedmont region.

 It may also be a modification of Huguenot French origin, probably a variant of "d'Aunay", of geographical origin.

 It may be also a Gaelic patronymic, since it is a common forename in Ireland.

 Dana is a relatively common surname in the US, ranking 7161 out of 88,799 in the 1990 U.S. Census.

 Dana is also the Persian word for wisdom.
</item 5>

<item 6>
Dana Ivey
Dana Robins Ivey (born August 12, 1941) is an American actress.

 She is a five-time Tony Award nominee for her work on Broadway, and won the 1997 Drama Desk Award for Outstanding Featured Actress in a Play for her work in
both "Sex and Longing" and "The Last Night of Ballyhoo".

 Her film appearances include "The Color Purple" (1985), "Dirty Rotten Scoundrels" (1988), "The Addams Family" (1991), "Two Weeks Notice" (2002), "Rush Hour 3"
(2007) and "The Help" (2011).
</item 6>

<item 7>
Two Weeks Notice
Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.

 The film was written and directed by Marc Lawrence, who had previously scripted two other box-office hits for Bullock, "Miss Congeniality" and "Forces of
Nature".

 Although critical response was mixed, the film received a successful box office run, both in the United States and globally.
</item 7>

<item 8>
Kathryn Drysdale (born December 1981) is an English actress known for Louise Brooks in *Two Pints of Lager and a Packet of Crisps* (2001–09), roles in *Vanity
Fair*, *Benidorm*, *Any Human Heart*, *Tripping Over*, and *Doctor Who*, and Meghan Markle in *The Windsors*. *These Old Broads* is a 2001 Carrie Fisher
television film starring Debbie Reynolds, Shirley MacL
</item 8>

<item 9>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 9>

<item 10>
Jaimee Foxworth (born 1979) is an American actress and model best known as Judy Winslow on *Family Matters*. She later appeared in several pornographic films
under the name Crave. Dana Ivey (born 1941) is an American actress and five-time Tony nominee. She won a 1997 Drama Desk Award and appeared in films including
*The Color Purple*, *The Addams Family*, *Two Weeks Notice* and *The Help*.
</item 10>
=== END ITEM id=1aeb3089a4dde0c1 ===

=== ITEM id=2c52159f5a13d865 ===
Gold sentences and facts:
Sentence 0: Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.
  Fact 0.0: Ranger Rick was originally called Ranger Rick's Nature Magazine
  Fact 0.1: Ranger Rick is a children's nature magazine
  Fact 0.2: Ranger Rick is published by the United States National Wildlife Federation
Sentence 1: TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.
  Fact 1.0: TENNIS is an American sports magazine
  Fact 1.1: TENNIS is owned by the Sinclair Broadcast Group

Retrieved items:
<item 1>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 1>

<item 2>
Tennis (magazine)
TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.

 There are eight issues per year covering the world of tennis.

 The "tennis.com" website is an offshoot of the magazine and part of the same media company.
</item 2>

<item 3>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 3>

<item 4>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 4>

<item 5>
Ranger Rick
Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.

 The magazine offers feature articles and activities for children, ages 7 and up, in order to spark their interest in the outdoors and become more actively
involved in the environment.

 The magazine's primary intention is to instill a passion for nature and promote activity outdoors.

 NWF also publishes two companion magazines, "Ranger Rick Jr.", which is aimed at ages 4–7, and "Ranger Rick Cub", which is aimed at kids 0–4 years old.)
</item 5>

<item 6>
Margaret Court
Margaret Court {'1': ", '2': ", '3': ", '4': "} (née Smith; born 16 July 1942), also known as Margaret Smith Court, is a retired Australian tennis player and
former world No. 1.

 She is currently a Christian minister in Perth, Western Australia.

 In tennis, she amassed more major titles than any other player in history.
</item 6>

<item 7>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 7>

<item 8>
Lorin Thompson
Lorin Thompson, Lorin Hartwell Jr Thompson, (1911–1997), was a muralist, artist, and creator and illustrator of the character "Ranger Rick" for the National
Wildlife Federation's children's magazine, Ranger Rick.
</item 8>

<item 9>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 9>

<item 10>
Lee–Lin rivalry
The Lee–Lin rivalry is between two legendary badminton players, Datuk Wira Lee Chong Wei of Malaysia and Lin Dan of China.

 The rivalry is considered to be one of, if not, the greatest rivalry in badminton history.

 More often than not, the dominance of the two legendary shuttlers have been compared to that of tennis greats Roger Federer and Rafael Nadal in tennis.

 They have played a total of 38 times, and Lin Dan leads their rivalry 26–12.

 Lee Chong Wei and Lin Dan are by far the two of the most dominant players across three generations and many regard Lin Dan and Lee Chong Wei as the 2 greatest
badminton players of all time.

 In their homeland, each hold a military rank, with Lee a Commander (Honorary) of the Royal Malaysian Navy Volunteer Reserve Unit and Lin a retired
Lieutenant-Colonel in the People's Liberation Army having being promoted from the ranks of Lieutenant-Commander and Major respectively, meaning they have the
same seniority in terms of military rank.
</item 10>
=== END ITEM id=2c52159f5a13d865 ===

=== ITEM id=1518e5fb36215121 ===
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
Czech presidential election, 1993
The 1993 Czech presidential election was held on 26 January 1993 to elect the first President of independent Czech Republic.

 The first election was done indirectly by the members of the Czech Republic Parliament.

 Václav Havel was elected President.

 The election was complicated only by attacks of Republicans against Havel and by bomb threat to the Parliament.
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
Remus Nicolai
Remus Nicolai (b. June 10, 1977 in Bistriţa, Romania) is a retired Romanian aerobic gymnast.

 He had a successful career winning six world championships medals (one gold, two silver and three bronze) and five European championships medals (two gold, two
silver and one bronze).

 After retiring from aerobic gymnastics he opened together with his wife (Daniela Mărănducă) a private gymnastics club in Constanţa.

 As of 2013 he trains, together with his wife, the junior artistic women's National Team of Romania at the National Olympic Center in Onesti.
</item 4>

<item 5>
Nicolae Titulescu
Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister,
and for two terms President of the General Assembly of the League of Nations (1930–32).
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
Ghimpețeni
Ghimpețeni is a commune in Olt County, Romania.

 It is composed of two villages, Ghimpețeni and Ghimpețenii Noi.

 These were part of Nicolae Titulescu Commune until 2004, when they were split off.
</item 7>

<item 8>
Gheorghe Tătărescu
Gheorghe I. Tătărescu (also known as "Guță Tătărescu", with a slightly antiquated pet form of his given name; 2 November 1886 – 28 March 1957) was a Romanian
politician who served 36th Prime Minister of Romania (1934–1937; 1939–1940), three times as Minister of Foreign Affairs ("interim" in 1934 and 1938; appointed
to the office in 1945-1947), and once as Minister of War (1934).

 Representing the "young liberals" faction inside the National Liberal Party (PNL), Tătărescu began his political career as a collaborator of Ion G. Duca,
becoming noted for his anti-Communism and, in time, for his conflicts with the PNL's leader Dinu Brătianu and the Foreign Minister Nicolae Titulescu.
</item 8>

<item 9>
Marius Urzică
Marius Daniel Urzică (born September 30, 1975, in Topliţa, Romania) is a Romanian gymnast.

 Urzică is an Olympic champion, a three-time world champion and a three-time European champion on pommel horse.

 He competed at three Olympic games, medaling each time on pommel horse (gold Sydney 2000, silver Atlanta 1996 and silver Athena 2004) and contributed to the
team bronze in Athens 2004.

 His unique technique and style of performance have won him the recognition as one of the greatest masters on this piece of apparatus ever, together with
Miroslav Cerar and Zoltán Magyar.

 Known as "The King of the Pommels" in 2001 he achieved the maximum score of 10.00 on this piece of apparatus at the Glasgow Grand Prix.

 Two elements in artistic gymnastics, one on pommel horse and one on parallel bars are named after him.
</item 9>

<item 10>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
</item 10>
=== END ITEM id=1518e5fb36215121 ===

=== ITEM id=9b8a67dceaaa1072 ===
Gold sentences and facts:
Sentence 0: Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.
  Fact 0.0: Alice Ann Munro's nee name is Laidlaw.
  Fact 0.1: Alice Ann Munro was born on 10 July 1931.
  Fact 0.2: Alice Ann Munro is a Canadian short story writer.
  Fact 0.3: Alice Ann Munro is a Nobel Prize winner.
Sentence 1: Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well as her left-wing sympathies and political activism.
  Fact 1.0: Lillian Florence Hellman was born on June 20, 1905.
  Fact 1.1: Lillian Florence Hellman died on June 30, 1984.
  Fact 1.2: Lillian Florence Hellman was an American dramatist.
  Fact 1.3: Lillian Florence Hellman was an American screenwriter.
  Fact 1.4: Lillian Florence Hellman is known for her success as a playwright on Broadway.
  Fact 1.5: Lillian Florence Hellman is known for her left-wing sympathies.
  Fact 1.6: Lillian Florence Hellman is known for her political activism.

Retrieved items:
<item 1>
The Little Foxes (film)
The Little Foxes (1941) is an American drama film directed by William Wyler.

 The screenplay by Lillian Hellman is based on her 1939 play of the same name.

 Hellman's ex-husband Arthur Kober, Dorothy Parker and her husband Alan Campbell contributed additional scenes and dialogue.
</item 1>

<item 2>
Julia (1977 film)
Julia is a 1977 American drama film directed by Fred Zinnemann, from a screenplay by Alvin Sargent.

 It is based on a chapter from Lillian Hellman's book "Pentimento" about the author's relationship with a lifelong friend, "Julia," who fought against the Nazis
in the years prior to World War II.

 Hellman said the story was true, but critics have challenged its accuracy.

 The film in DeLuxe Color was produced by Richard Roth, with Julien Derode as executive producer and Tom Pevsner as associate producer.
</item 2>

<item 3>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 3>

<item 4>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 4>

<item 5>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 5>

<item 6>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 6>

<item 7>
Anaïs Nin
Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban
parents in France, where she was also raised.

 She spent some time in Spain and Cuba, but lived most of her life in the United States, where she became an established author.

 She wrote journals (which span more than 60 years, beginning when she was 11 years old and ending shortly before her death), novels, critical studies, essays,
short stories, and erotica.

 A great deal of her work, including "Delta of Venus" and "Little Birds", was published posthumously.
</item 7>

<item 8>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 8>

<item 9>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 9>

<item 10>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 10>
=== END ITEM id=9b8a67dceaaa1072 ===

=== ITEM id=8eb1bef292a7a01e ===
Gold sentences and facts:
Sentence 0: Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.
  Fact 0.0: Eric Arthur Blair was born on 25 June 1903.
  Fact 0.1: Eric Arthur Blair died on 21 January 1950.
  Fact 0.2: Eric Arthur Blair's pen name was George Orwell.
  Fact 0.3: Eric Arthur Blair is better known as George Orwell.
  Fact 0.4: Eric Arthur Blair was an English novelist.
  Fact 0.5: Eric Arthur Blair was an English essayist.
  Fact 0.6: Eric Arthur Blair was an English journalist.
  Fact 0.7: Eric Arthur Blair was an English critic.
Sentence 1: Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.
  Fact 1.0: Nelson Richard DeMille was born on August 23, 1943.
  Fact 1.1: Nelson Richard DeMille is an American author.
  Fact 1.2: Nelson Richard DeMille writes action adventure novels.
  Fact 1.3: Nelson Richard DeMille writes suspense novels.
Sentence 2:  DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
  Fact 2.0: Nelson Richard DeMille has written under the pen name Jack Cannon.
  Fact 2.1: Nelson Richard DeMille has written under the pen name Kurt Ladner.
  Fact 2.2: Nelson Richard DeMille has written under the pen name Ellen Kay.
  Fact 2.3: Nelson Richard DeMille has written under the pen name Brad Matthews.

Retrieved items:
<item 1>
The Panther (novel)
The Panther is a 2012 novel by American author Nelson DeMille.

 It is the sixth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "The Lion".

 The Panther is followed by DeMille’s 2015 novel, "Radiant Angel".

 Also featured in this novel is DeMille’s other fictional character, Paul Brenner, who appears in The General's Daughter and Up Country.
</item 1>

<item 2>
George Orwell: A Life in Pictures
George Orwell: A Life in Pictures is a 2003 BBC Television docudrama telling the life story of the British author George Orwell.

 Chris Langham plays the part of Orwell.

 No surviving sound recordings or video of the real George Orwell have been found.
</item 2>

<item 3>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 3>

<item 4>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 4>

<item 5>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 5>

<item 6>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 6>

<item 7>
George Orwell
Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.

 His work is marked by lucid prose, awareness of social injustice, opposition to totalitarianism, and outspoken support of democratic socialism.
</item 7>

<item 8>
Wild Fire (novel)
Wild Fire is a 2006 novel by American author, Nelson DeMille.

 It is the fourth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to Night Fall and takes place approximately one year later.

 Wild Fire is followed by DeMille’s 2010 novel, The Lion.
</item 8>

<item 9>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 9>

<item 10>
The Lion's Game
The Lion's Game is a 2000 novel by American author Nelson DeMille.

 It is the second of DeMille's novels to feature the detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New
York City.

 The 2000 novel The Lion’s Game is the sequel to Plum Island and is followed by the 2004 book Night Fall.

 The book also briefly mentions events from other DeMille novels like "The Charm School" and "The Gold Coast", that aren't strictly part of the Corey-universe.
</item 10>
=== END ITEM id=8eb1bef292a7a01e ===

=== ITEM id=08fe904fa926a78c ===
Gold sentences and facts:
Sentence 0: Sasson Gabai (Hebrew: ששון גבאי‎ ‎ ; born 24 November 1947) is an Israeli actor.
  Fact 0.0: Sasson Gabai's Hebrew name is ששון גבאי.
  Fact 0.1: Sasson Gabai was born on 24 November 1947.
  Fact 0.2: Sasson Gabai is an Israeli actor.
Sentence 1:  Winner of an Ophir Award, Asia Pacific Screen Award (nominated), European Film Awards and Jerusalem Film Festival in 2007 as Best Actor in a highly acclaimed Israeli film "The Band's Visit".
  Fact 1.0: Sasson Gabai won an Ophir Award.
  Fact 1.1: Sasson Gabai was nominated for an Asia Pacific Screen Award.
  Fact 1.2: Sasson Gabai won a European Film Awards award.
  Fact 1.3: Sasson Gabai won a Jerusalem Film Festival award.
  Fact 1.4: These awards were won in 2007.
  Fact 1.5: The awards were for Best Actor.
  Fact 1.6: The award-winning role was in the highly acclaimed Israeli film The Band's Visit.
Sentence 2: The Band's Visit (Hebrew: ביקור התזמורת - "Bikur Ha-Tizmoret") is a 2007 Israeli film directed by Eran Kolirin.
  Fact 2.0: The Band's Visit is a 2007 Israeli film.
  Fact 2.1: The Band's Visit is written in Hebrew as ביקור התזמורת.
  Fact 2.2: The Hebrew title of The Band's Visit is Bikur Ha-Tizmoret.
  Fact 2.3: The Band's Visit was directed by Eran Kolirin.

Retrieved items:
<item 1>
Beyond the Sea (1991 film)
Beyond the Sea (translit.

 Me'ever Layam) is a 1991 Israeli drama film directed by Jacob Goldwasser.

 It stars Aryeh Moskona, Dafna Rechter, Moti Giladi.

 The film, a family drama set in the 1960s, was critically acclaimed, and won the Ophir Award for Best Film.

 The film was selected as the Israeli entry for the Best Foreign Language Film at the 64th Academy Awards, but was not accepted as a nominee.
</item 1>

<item 2>
Sh'Chur
Sh'Chur is a 1994 Israeli drama film starring Gila Almagor, Ronit Elkabetz and Hanna Azoulay-Hasfari.

 It was written by Hanna Azoulay Hasfari and directed by her partner Shmuel Hasfari.

 Sh'Chur received critical acclaim and was the 1994 official Israeli submission for the Academy Award for Best Foreign Language Film.

 It was also awarded the Ophir Award for best film by the Israeli Film and Television Academy.

 After its release, the film garnered various discussions in the Israeli press over its representation of the Moroccan community in Israel.
</item 2>

<item 3>
The Band's Visit
The Band's Visit (Hebrew: ביקור התזמורת - "Bikur Ha-Tizmoret") is a 2007 Israeli film directed by Eran Kolirin.
</item 3>

<item 4>
Foxtrot (2017 film)
Foxtrot (Hebrew: פוֹקְסטְרוֹט‎ ‎ ) is a 2017 Israeli drama film directed by Samuel Maoz.

 It was screened in the competition section of the 74th Venice International Film Festival where it won the Grand Jury Prize Silver Lion.

 It was also screened in the Special Presentations section at the 2017 Toronto International Film Festival.

 It won the Ophir Award for Best Film, therefore becoming the Israeli entry for the Best Foreign Language Film at the 90th Academy Awards.
</item 4>

<item 5>
Joseph Cedar
Yossef (Joseph) Cedar (Hebrew: יוסף סידר; born August 31, 1968) is an Israeli film director and screenwriter.

 He has won a Silver Bear and an Ophir Award for Best Director, and an Ophir Award for writing a Best Screenplay.

 He also won the best screenplay award at the 2011 Cannes Film Festival for his film "Footnote" (2011).
</item 5>

<item 6>
Yana's Friends
Yana's Friends (Hebrew: החברים של יאנה, "HaHaverim shel Yana"‎ ‎ ) is a 1999 Israeli film directed by Arik Kaplun.

 script editor: Savi Gabizon.

 Critically acclaimed, it won 10 Israeli Academy Awards including the Ophir Award for Best Picture.

 It also won the Crystal Globe at the Karlovy Vary International Film Festival in 1999.

 The film has a very rare 100% rating on the film website Rotten Tomatoes based on 30 reviews.
</item 6>

<item 7>
The Farewell Party
The Farewell Party (Hebrew: מיתה טובה‎ ‎ Mita Tova) is a 2014 Israeli drama film about the use of a euthanasia device, directed by Tal Granit and Sharon Maymon.

 It was nominated for the Ophir Award for Best Film.

 The film was screened in the Venice Days section of the 71st Venice International Film Festival and has been selected to be screened in the Contemporary World
Cinema section at the 2014 Toronto International Film Festival.

 The film was also screened in the 44th edition of the International Film Festival Rotterdam, reaching the second place in the IFFR audience award.
</item 7>

<item 8>
Hunting Elephants
Hunting Elephants (Hebrew: לצוד פילים‎ ‎ , "Latzud Pilim", "lit."

 To Hunt Elephants) is a 2013 Israeli crime comedy film directed by Reshef Levi.

 It stars, among others, Sasson Gabai, Moni Moshonov and Patrick Stewart.

 It was released to theaters in Israel on July 4, 2013.
</item 8>

<item 9>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 9>

<item 10>
Sasson Gabai
Sasson Gabai (Hebrew: ששון גבאי‎ ‎ ; born 24 November 1947) is an Israeli actor.

 Winner of an Ophir Award, Asia Pacific Screen Award (nominated), European Film Awards and Jerusalem Film Festival in 2007 as Best Actor in a highly acclaimed
Israeli film "The Band's Visit".
</item 10>
=== END ITEM id=08fe904fa926a78c ===

=== ITEM id=0289611b7539b523 ===
Gold sentences and facts:
Sentence 0: Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.
  Fact 0.0: Karen Joy Fowler was born on February 7, 1950.
  Fact 0.1: Karen Joy Fowler is American.
  Fact 0.2: Karen Joy Fowler is an author of science fiction.
  Fact 0.3: Karen Joy Fowler is an author of fantasy.
  Fact 0.4: Karen Joy Fowler is an author of literary fiction.
Sentence 1: Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.
  Fact 1.0: Charles Bruce Chatwin was born on 13 May 1940.
  Fact 1.1: Charles Bruce Chatwin died on 18 January 1989.
  Fact 1.2: Charles Bruce Chatwin was English.
  Fact 1.3: Charles Bruce Chatwin was a travel writer.
  Fact 1.4: Charles Bruce Chatwin was a novelist.
  Fact 1.5: Charles Bruce Chatwin was a journalist.

Retrieved items:
<item 1>
The Jane Austen Book Club
The Jane Austen Book Club is a 2004 novel by American author Karen Joy Fowler.

 The story, which takes place near Sacramento, California, centers on a book club consisting of five women and one man who meet once a month to discuss Jane
Austen's six novels.

 It was a critical success and became a national bestseller.
</item 1>

<item 2>
Graph query row:
person: Karen Joy Fowler
birth_date: February 7, 1950
connected_person: Bruce Chatwin
connected_birth_date: 13 May 1940
connection: None
</item 2>

<item 3>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 3>

<item 4>
Jules Wright
Jules Wright (25 February 1948 – 21 June 2015) was an Australian-born theatre director, a co-founder in 1984 of the Women’s Playhouse Trust (WPT), the first
resident woman director at the Royal Court Theatre, being only the second woman to direct on its main stage, and founder of the Wapping Project arts space.
</item 4>

<item 5>
Bruce Chatwin
Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.

 His first book, "In Patagonia" (1977), established Chatwin as a travel writer, although he considered himself instead a storyteller, interested in bringing to
light unusual tales.

 He won the James Tait Black Memorial Prize for his novel "On the Black Hill" (1982) and his novel "Utz" (1988) was shortlisted for the Man Booker Prize.

 In 2008 "The Times" named Chatwin #46 on their list of "50 Greatest British Writers Since 1945."
</item 5>

<item 6>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 6>

<item 7>
We Are All Completely Beside Ourselves
We Are All Completely Beside Ourselves is a 2013 novel by the American writer Karen Joy Fowler.

 The novel won the 2014 PEN/Faulkner Award for Fiction<ref name="The Guardian - 3 April 2014 - Karen Joy Fowler wins PEN/Faulkner award for fiction"> </ref> and
was also short-listed for the 2014 Man Booker Prize.
</item 7>

<item 8>
Karen Joy Fowler
Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.

 Her work often centers on the nineteenth century, the lives of women, and alienation.
</item 8>

<item 9>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 9>

<item 10>
Always (short story)
"Always" is a science fiction short story written by Karen Joy Fowler.

 Originally published in the April/May 2007 issue of "Asimov's Science Fiction", it won the 2007 Nebula Award for Best Short Story.

 It is collected in Fowler's "What I Didn't See".
</item 10>
=== END ITEM id=0289611b7539b523 ===

=== ITEM id=7d838bd04f2ed7b8 ===
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
Grand Forks International Airport
Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County,
North Dakota.

 GFK has no scheduled passenger flights out of the country but has an "international" title (like many other airports) because it has customs service for
arrivals from Canada and other countries.
</item 1>

<item 2>
Downtown Grand Forks
Downtown Grand Forks is the original commercial center of Grand Forks, North Dakota.

 Located on the western bank of the Red River of the North, the downtown neighborhood is situated near the fork of the Red River and the Red Lake River.

 While downtown is no longer the dominant commercial area of the Greater Grand Forks community, it remains the historic center of Grand Forks.

 An 80.4 acre portion was listed on the National Register of Historic Places in 2005, as Downtown Grand Forks Historic District.

 Today, downtown Grand Forks is home to many offices, stores, restaurants, and bars.
</item 2>

<item 3>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 3>

<item 4>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 4>

<item 5>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 5>

<item 6>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 6>

<item 7>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
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
Emerado, North Dakota
Emerado is a city in Grand Forks County, North Dakota, United States located near Grand Forks Air Force Base.

 It is part of the "Grand Forks, ND-MN Metropolitan Statistical Area" or "Greater Grand Forks".

 The population was 414 at the 2010 census.

 Grand Forks Air Force Base is located near Emerado.
</item 9>

<item 10>
Grand Forks International
The Grand Forks International (GFI) is an annual international invitational baseball tournament hosted at James Donaldson Park in Grand Forks, British Columbia.

 The GFI is the largest invitational baseball tournament in Canada and is a large part of the culture in Grand Forks.

 The tournament is also notable for being run entirely by volunteers.
</item 10>
=== END ITEM id=7d838bd04f2ed7b8 ===

=== ITEM id=225ba2d2f0b7ca52 ===
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
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 1>

<item 2>
Henry James
Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary
modernism, and is considered by many to be among the greatest novelists in the English language.

 He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
</item 2>

<item 3>
The Frost King
"The Frost King" was a short story about King Jack Frost written by 11-year-old Helen Keller.

 Keller's teacher, Anne Sullivan, had mentioned that the autumn leaves were "painted ruby, emerald, gold, crimson, and brown," and Keller, by her own account,
imagined fairies doing the work.

 Keller wrote a story about how a cask of jewels, being transported by fairy servants, had melted in the sun and covered the leaves.

 As a birthday gift, Keller sent the story to Michael Anagnos, head of the Perkins School for the Blind, who published the story in "The Mentor," the Perkins
alumni magazine.

 It was picked up by "The Goodson Gazette", a journal on deaf-blind education, based in Virginia.
</item 3>

<item 4>
James Poe
James Poe (October 4, 1921 – January 24, 1980) was an American film and television screenwriter.

 He is best known for his work on the movies "Around the World in 80 Days" for which he jointly won an Academy Award in 1956, "Cat on a Hot Tin Roof", "Summer
and Smoke", "Lilies of the Field", and "They Shoot Horses, Don't They?

".
</item 4>

<item 5>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 5>

<item 6>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 6>

<item 7>
A Predicament
"A Predicament" is a humorous short story by Edgar Allan Poe, usually combined with its companion piece "How to Write a Blackwood Article."

 It was originally titled "The Scythe of Time".

 The paired stories parody the Gothic sensation tale, popular in England and America since the early 19th century.
</item 7>

<item 8>
Anne Perry
Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk
series.

 At the age of fifteen she was convicted of participating in the murder of her friend's mother, in 1954.

 She changed her name after serving her five-year sentence.
</item 8>

<item 9>
Haunted (Poe album)
Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in
1995.

 The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
</item 9>

<item 10>
The Spectre Knight
The Spectre Knight is a one-act "fanciful operetta" with a libretto by James Albery and music by Alfred Cellier.

 It was first performed on 9 February 1878 at the Opera Comique by the Comedy Opera Company as a companion piece to "The Sorcerer".

 The piece continued to run until 23 March 1878 and was revived by the company from 28 May 1878 to 10 August 1878 as a companion piece to "H.M.S. Pinafore".

 The piece had a run in New York in 1880 and was toured in Britain and America.
</item 10>
=== END ITEM id=225ba2d2f0b7ca52 ===
