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

This file has 20 items.

=== ITEM id=a088-71b3651e15fe35ca ===
Gold sentences and facts:
Sentence 0:  He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West Tonight" from January 1997 to October 2011.
  Fact 1.0: The person referred to as 'He' was the host of 'The Krypton Factor' for its original 18-year run.

Retrieved items:
<item 1>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 1>

<item 2>
The Krypton Factor
The Krypton Factor is a British game show produced by Granada for broadcast on ITV.

 The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
</item 2>

<item 3>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 3>

<item 4>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 4>

<item 5>
Password (UK game show)
Password was a British panel game show based on the US version of the same name.

 It was originally aired on ITV produced by ATV from 12 March to 10 September 1963 hosted by Shaw Taylor, then it aired on BBC2 from 24 March to 28 April 1973
hosted by Brian Redhead before moving to its flagship channel BBC1 from 7 January 1974 to 1976 first hosted by Eleanor Summerfield then by Esther Rantzen, it
was then aired on Channel 4 produced by Thames in association with Talbot Television and Goodson-Todman Productions from 6 November 1982 to 14 May 1983 hosted
by Tom O'Connor and then finally aired back on ITV produced by Ulster from 22 July 1987 to 5 August 1988 hosted by Gordon Burns.
</item 5>

<item 6>
David Letterman
David Michael Letterman (born April 12, 1947) is an American television host, comedian, writer, and producer.

 He hosted a late night television talk show for 33 years, beginning with the February 1, 1982 debut of "Late Night with David Letterman" on NBC, and ending
with the May 20, 2015 broadcast of "Late Show with David Letterman" on CBS.

 In total, Letterman hosted 6,028 episodes of "Late Night" and "Late Show", surpassing friend and mentor Johnny Carson as the longest-serving late night talk
show host in American television history.

 In 1996 Letterman was ranked 45th on "TV Guide"' s 50 Greatest TV Stars of All Time.
</item 6>

<item 7>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 7>

<item 8>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 8>

<item 9>
Gordon Burns (footballer)
Gordon Burns (born 2 December 1978) is a Scottish former professional footballer who is currently manager of Troon in the Scottish Junior Football Association,
West Region.

 He has previously played in the Scottish Championship for Ayr United.
</item 9>

<item 10>
Gordon Burns
Gordon Henry Burns (born 10 June 1942) is a Northern Irish journalist and broadcaster.

 He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West
Tonight" from January 1997 to October 2011.

 In November 2011 Gordon moved back to Belfast were he was born.
</item 10>
=== END ITEM id=a088-71b3651e15fe35ca ===

=== ITEM id=a085-26b735a246658695 ===
Gold sentences and facts:
Sentence 0: Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and Randall Counties, Texas, United States.
  Fact 0.4: Rick Husband Amarillo International Airport is six miles (10 km) east of downtown Amarillo.

Retrieved items:
<item 1>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
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
Brainerd Lakes Regional Airport is a city- and county-owned, general-aviation airport near Brainerd, Minnesota, with one commercial airline. Grand Sky,
established in 2015 at Grand Forks Air Force Base, is a 217-acre UAS business park with runway access. Grand Forks Municipal Airport closed after World War II
and became urban land; Grand Forks International Airport serves the area and offers customs but no scheduled international flights. Nearby Emerado had 414
residents in
</item 4>

<item 5>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 5>

<item 6>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 6>

<item 7>
Emerado, North Dakota
Emerado is a city in Grand Forks County, North Dakota, United States located near Grand Forks Air Force Base.

 It is part of the "Grand Forks, ND-MN Metropolitan Statistical Area" or "Greater Grand Forks".

 The population was 414 at the 2010 census.

 Grand Forks Air Force Base is located near Emerado.
</item 7>

<item 8>
Downtown Grand Forks
Downtown Grand Forks is the original commercial center of Grand Forks, North Dakota.

 Located on the western bank of the Red River of the North, the downtown neighborhood is situated near the fork of the Red River and the Red Lake River.

 While downtown is no longer the dominant commercial area of the Greater Grand Forks community, it remains the historic center of Grand Forks.

 An 80.4 acre portion was listed on the National Register of Historic Places in 2005, as Downtown Grand Forks Historic District.

 Today, downtown Grand Forks is home to many offices, stores, restaurants, and bars.
</item 8>

<item 9>
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
</item 9>

<item 10>
Grand Forks International
The Grand Forks International (GFI) is an annual international invitational baseball tournament hosted at James Donaldson Park in Grand Forks, British Columbia.

 The GFI is the largest invitational baseball tournament in Canada and is a large part of the culture in Grand Forks.

 The tournament is also notable for being run entirely by volunteers.
</item 10>
=== END ITEM id=a085-26b735a246658695 ===

=== ITEM id=a106-8b68a8a5c6e79488 ===
Gold sentences and facts:
Sentence 0: Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.
  Fact 0.4: Christopher Edward Nolan is a screenwriter

Retrieved items:
<item 1>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 1>

<item 2>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 2>

<item 3>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 3>

<item 4>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 4>

<item 5>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 5>

<item 6>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 6>

<item 7>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 7>

<item 8>
Blue Collar (film)
Blue Collar is a 1978 American crime drama film directed by Paul Schrader, in his directorial debut.

 It was written by Schrader and his brother Leonard, and stars Richard Pryor, Harvey Keitel and Yaphet Kotto.
</item 8>

<item 9>
The Walker
The Walker is a 2007 American-British drama film written and directed by Paul Schrader.

 It is an independent production and is the latest installment in Schrader's "night workers" series of films, starting with "Taxi Driver" in 1976, followed by
"American Gigolo" in 1980 and "Light Sleeper" in 1992.
</item 9>

<item 10>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 10>
=== END ITEM id=a106-8b68a8a5c6e79488 ===

=== ITEM id=a077-9c0c47f313db40d8 ===
Gold sentences and facts:
Sentence 0:  One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless charisma and self-assurance.
  Fact 1.0: Dean Martin was one of the most popular and enduring American entertainers of the mid-20th century.

Retrieved items:
<item 1>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 1>

<item 2>
Basie Jazz
Basie Jazz is an album by pianist/bandleader Count Basie recorded in 1952 and released on the Clef label in 1954.

 Selections from this album were also released on the 1956 Clef LPs "The Swinging Count!

" and "Basie Rides Again!

".
</item 2>

<item 3>
Swingin' Down Yonder
Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and
February 1955.

 According to the original sleeve notes, all the songs have a "common geographical root: the American South."

 In 1963, Capitol Records re-released "Swingin' Down Yonder" under the titled "Southern Style".
</item 3>

<item 4>
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
</item 4>

<item 5>
Things Ain't What They Used to Be
In 1941 there was a strike against the American Society of Composers, Authors and Publishers, of which Duke Ellington was a member.

 Because of the strike he could not air his songs on the radio.

 Instead, he used songs written by his son Mercer and pianist Billy Strayhorn.

 Strayhorn's compositions of this time include "Take the 'A' Train", "Chelsea Bridge" and "Day Dream", and Mercer wrote "Things Ain't What They Used to Be",
"Blue Serge" and "Moon Mist".
</item 5>

<item 6>
Chico Hamilton Trio
Chico Hamilton Trio is an album by drummer and bandleader Chico Hamilton recorded at sessions in 1953, 1954 and 1956 released on the Pacific Jazz label.

 The album features Hamilton's first recordings for Pacific Jazz from 1953 and 1954, six tracks originally released on a 10-inch album, along with an additional
four recordings from 1956.
</item 6>

<item 7>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 7>

<item 8>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 8>

<item 9>
One O'Clock Jump
"One O'Clock Jump" is a jazz standard, a 12-bar blues instrumental, written by Count Basie in 1937.

 The melody derived from band members' riffs--Basie rarely wrote down musical ideas, so Eddie Durham and Buster Smith helped him crystallize his ideas.

 The original 1937 recording of the tune by Basie and his band is noted for the saxophone work of Herschel Evans and Lester Young, trumpet by Buck Clayton,
Walter Page on bass and Basie himself on piano.
</item 9>

<item 10>
Cecil Gant
Cecil Gant (April 4, 1913 – February 4, 1951) was an American blues singer, songwriter and pianist, whose recordings of both ballads and "fiery piano rockers"
were successful in the mid- and late 1940s, and influenced the early development of rock and roll.

 His biggest hit was the 1944 ballad, "I Wonder".
</item 10>
=== END ITEM id=a077-9c0c47f313db40d8 ===

=== ITEM id=a100-329c006eda769187 ===
Gold sentences and facts:
Sentence 0: Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he founded in 2006.
  Fact 1.5: WikiLeaks is an organisation.

Retrieved items:
<item 1>
Micom Film and TV Productions
MICOM Film and TV Productions was established in 1989 by Sreco Mihelcic in a Former Yugoslavian republic Slovenia.

 In the early 90' company joined two sons, Matjaz Mihelcic - Program director, Producer and Vasja Mihelcic - technical director and Executive Producer.

 MICOM was in 1990s the leading Slovenian creative house, producing TV shows for youth, popular entertainment TV shows and discovered many new TV concepts.

 Micom discovered many talents, different creative people and gave them possibility to express themselves.

 The most popular TV shows were: Lahkih nog naokrog (educational youth TV series), Cari zacimb (entertaining cooking TV series), Razpoke v casu (fantasy TV
series).

 They also produced documentary movies.

 One of them got in 1996 internationally release - Fight for the river and it's distributed by International Historic Films.

.

 In the beginning of 2000 company joined Marjana Mihelcic.

 It was a family based company with long tradition in show and entertainment business.
</item 1>

<item 2>
Profiles span journalists, artists, politicians, scholars and public figures: Nikola Mladenov founded Macedonia’s respected Fokus and inspired an
investigative-journalism award; Johannes Grenzfurthner leads art group monochrom; François Genoud financed Nazi networks and Middle Eastern militants; Edmund de
Unger collected Islamic art; Adel Ferdosipour hosts Iran’s Navad; Hans Aschenborn painted African wildlife; Katarina Ivanovska models and acts. Nicolae
Titulescu
</item 2>

<item 3>
The text surveys diverse religious, political, civic, and governmental groups. The Atman Foundation, a Brahma Kumaris offshoot, faced a false collective-suicide
scare in Tenerife. Montenegro’s 2006 “Eagle’s Flight” operation arrested 17 alleged militants. Greece’s anarchist Conspiracy of Fire Cells carried out bombings;
the Indian National Council supported independence efforts. Other entries cover Sahara-Sahel environmental work, a declining Indian Trotskyist group,
Arab-American civil rights, diplomacy,
</item 3>

<item 4>
Micom Film and TV Productions
It was a family based company with long tradition in show and entertainment business.

 Micom produced around 30.000 minutes of program for National TV Slovenia and other media.

 In 2002 company stopped producing because of the tragedy in the family.

 Both sons went separate way and build their own Film and TV production house.
</item 4>

<item 5>
Profiles span companies, organizations, people, and technologies: Pharma Medica conducts pharmaceutical trials in Ontario and Missouri; Genpact is a global
BPM/IT firm, formerly led by Pramod Bhasin; Summit BHC operates U.S. addiction-treatment centers. Triple Canopy merged with Academi in 2014 to form Constellis.
Other subjects include geothermal binary-cycle power, FuelCell Energy, packaging maker Krones, Polish firms, media and investment groups, retailers,
</item 5>

<item 6>
Music West Records
Music West Records was an independent record company founded by Allan Kaplan on December 1985 in San Rafael, California.

 The company was initially formed to promote Ray Lynch.

 During its run, artists released under the record company included Jim Chappell, Kenneth Nash, Chris Spheeris, and Øystein Sevåg.

 According to Gary Chappell, the manufacturer for Music West, the artists originated independently, claiming that the company's idea "has a statement that comes
directly from the
</item 6>

<item 7>
Low Transit Industries
Low Transit Industries was an independent record label founded in Melbourne, Australia in 1999 by Darren Smallman (BATTLE WORLDWIDE) and Simon Baird.

 The label toured and released bands across the world until 2010 when the partners closed operations due to Darren Smallman moving to the UK where he now runs
BATTLE WORLDWIDE which is a record label, book publisher, music publisher and PR services company.
</item 7>

<item 8>
Admir Čavalić
Admir Čavalić is the Director of Association "Multi", a non-profit organization dedicated to promoting libertarian ideals in Bosnia and Herzegovina.

 "Multi" is the first non-profit organization registered in Bosnia and Herzegovina that advocates libertarian ideas.

 Čavalić is the director of OPEN Fest, biggest libertarian festival in Europe.

 Čavalić also works at Faculty of Economics University of Tuzla, as an assistant teacher.

 Čavalić with Dženan Smajić wrote "Islam and Free Market" in 2014.

 The book, that was published in Bosnian, provides arguments for Islam's connection with the ideas of economic liberalism.

 Čavalić made first bosnian documentary about economics, titled "Economy".
</item 8>

<item 9>
Mediastan
Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca
O'Brien and Lauren Dark.
</item 9>

<item 10>
Pharma Medica
Pharma Medica Research Inc. is an American research and development company that works with pharmaceutical companies and performs clinical trials in the
pharmaceutical and biotechnology industry.

 The company is based in Toronto, Canada with headquarters in Mississauga, Ontario.

 Pharma Medica has locations in Scarborough, Ontario and St. Charles, Missouri.
</item 10>
=== END ITEM id=a100-329c006eda769187 ===

=== ITEM id=a124-ff4ad07848c4134a ===
Gold sentences and facts:
Sentence 0: Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.
  Fact 0.2: Margaret Landon was an American writer.

Retrieved items:
<item 1>
Anna and the King of Siam (novel)
Anna and the King is a 1944 semi-fictionalized biographical novel by Margaret Landon.
</item 1>

<item 2>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 2>

<item 3>
Anna and the King of Siam (film)
Anna and the King of Siam is a 1946 drama film directed by John Cromwell.

 An adaptation of the 1944 novel of the same name by Margaret Landon, it was based on the fictionalized diaries of Anna Leonowens, an Anglo-Indian woman who
claimed to be British and became governess in the Royal Court of Siam (now modern Thailand) during the 1860s.

 Darryl F. Zanuck read Landon's book in galleys and immediately bought the film rights.
</item 3>

<item 4>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 4>

<item 5>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 5>

<item 6>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 6>

<item 7>
Margaret Landon
Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of
the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.

 In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.

 A later work, "Never Dies the Dream", appeared in 1949.
</item 7>

<item 8>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 8>

<item 9>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 9>

<item 10>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 10>
=== END ITEM id=a124-ff4ad07848c4134a ===

=== ITEM id=a078-6e40b089c813d766 ===
Gold sentences and facts:
Sentence 0: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 1.1: Chowder was created by C. H. Greenblatt

Retrieved items:
<item 1>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 1>

<item 2>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 2>

<item 3>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 3>

<item 4>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 4>

<item 5>
Harvey Beaks
Harvey Beaks is an American animated television series created by C.H. Greenblatt for Nickelodeon.

 The series premiered on March 28, 2015 following the 2015 Kids' Choice Awards.
</item 5>

<item 6>
Adventure Time
Adventure Time is an American animated television series created by Pendleton Ward for Cartoon Network.

 It follows the adventures of a boy named Finn (voiced by Jeremy Shada) and his best friend and adoptive brother Jake (John DiMaggio)— a dog with the magical
power to change shape and size at will.

 Finn and Jake live in the post-apocalyptic Land of Ooo, where they interact with Princess Bubblegum (Hynden Walch), the Ice King (Tom Kenny), Marceline the
Vampire Queen (Olivia Olson), BMO (Niki Yang), and others.

 The series is based on a 2007 short produced for Nicktoons and Frederator Studios' animation incubator series "Random!

 Cartoons".

 After the short became a viral hit on the Internet, Cartoon Network commissioned a full-length series, which previewed on March 11, 2010, and officially
premiered on April 5, 2010.
</item 6>

<item 7>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 7>

<item 8>
Sherry Alberoni
Sharyn Eileen "Sherry" Alberoni (born December 4, 1946) is an American actress and voice artist.

 Alberoni got her start as a Mouseketeer on the weekday ABC television program "The Mickey Mouse Club".

 As an adult, she became a voice artist for Hanna-Barbera Productions.

 Besides providing voices for numerous incidental characters in series such as "Jeannie", Alberoni is best known as the voice of nasty rich-girl Alexandra Cabot
from "Josie and the Pussycats" , "superhero-in-training" Wendy from the first season of "Super Friends", the heroic robot, Bo in "Mighty Orbots" and
Glumdalclitch in "The Three Worlds of Gulliver".

 In 1971, she starred alongside Patty Andrews in the Sherman Brothers stage musical, "Victory Canteen".
</item 8>

<item 9>
Adventure Time (pilot)
"Adventure Time" is an animated short created by Pendleton Ward, as well as the pilot to the Cartoon Network series of the same name.

 The short follows the adventures of Pen (voiced by Zack Shada), a human boy, and his best friend Jake (voiced by John DiMaggio), a dog with magical powers to
change shape and grow and shrink at will.

 In this episode, Pen and Jake have to rescue Princess Bubblegum (voiced by Paige Moss) from the antagonistic Ice King (voiced by John Kassir).
</item 9>

<item 10>
Right Now Kapow
Right Now Kapow is an American animated sketch comedy television series produced by Warner Bros.

 Animation for Disney XD.

 It is the first collaboration between Warner Bros.

 Animation and Disney.

 The series premiered on September 19, 2016 and ended on May 31, 2017.

 The series was created by Justin Becker and Marly Halpern-Graser, who previously worked on the Cartoon Network series "Mad".

 Becker also worked on Adult Swim infomercials, and Halpern-Graser also worked on Cartoon Network's "DC Nation".

 The series follows Dog, Candy, Ice Cream, Diamond, Plant, and Moon going on new adventures everyday...and find themselves in mischief.

 Every episode of the series has 3 main parts in every 11-minute segment and other random shorts.

 The series has a style of humor similar to Cartoon Network's previous series, "Mad".
</item 10>
=== END ITEM id=a078-6e40b089c813d766 ===

=== ITEM id=a092-3fb5735c13e1b7fd ===
Gold sentences and facts:
Sentence 0:  She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight Rises".
  Fact 1.4: The actress has appeared in the film Maleficent.

Retrieved items:
<item 1>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 1>

<item 2>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 2>

<item 3>
Dirty Girl (2010 film)
Dirty Girl is a 2010 American coming of age comedy-drama film written and directed by Abe Sylvia.

 It stars Juno Temple, Milla Jovovich and William H. Macy.

 It premiered at the 2010 Toronto International Film Festival on September 12, 2010.

 It was distributed theatrically by The Weinstein Company on October 7, 2011.
</item 3>

<item 4>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 4>

<item 5>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
</item 5>

<item 6>
Kate Winslet, a British actress and singer, debuted as Juliet Hulme in *Heavenly Creatures* (1994). Her early films included *Sense and Sensibility* (1995),
earning a Best Supporting Actress Oscar nomination; *Jude* and *Hamlet* (1996); and *Titanic* (1997), which brought Best Actress nominations from the Academy,
Golden Globe, and Screen Actors Guild Awards.
</item 6>

<item 7>
Juliet Weber works in the film industry as a producer, director, writer, and editor. She co-produced *Fastpitch* (2000) and *Unchained Memories* (2003).
</item 7>

<item 8>
The entries profile performers and productions, including *Doctor Who*, *EastEnders* and *Waterloo Road*. Juliet Weber is a producer, director, writer and
editor. Juno Temple starred in *Vinyl*; Holly Hunter won major awards for *The Piano*. Kathryn Drysdale played Louise Brooks and Meghan Markle. Kate Winslet
rose to prominence with *Heavenly Creatures* and *Titanic*. Cate Blanchett won two Oscars, including for *The Aviator*. Reese
</item 8>

<item 9>
Cracks (film)
Cracks is a 2009 independent drama psychological thriller film starring Eva Green, James McGovern, Juno Temple, María Valverde, and Imogen Poots, which was
released theatrically in the UK and Ireland on 4 December 2009.

 In the United States it was released by IFC Films on 18 March 2011, and premiered on Showtime in late 2011.
</item 9>

<item 10>
Juno Temple (born 21 July 1989) is an English actress known for films including *Black Mass*, *The Other Boleyn Girl*, *Wild Child*, *Atonement*, *Maleficent*,
*The Three Musketeers* and *The Dark Knight Rises*. She starred in HBO’s period drama *Vinyl*.
</item 10>
=== END ITEM id=a092-3fb5735c13e1b7fd ===

=== ITEM id=a003-72eb50561fbeb609 ===
Gold sentences and facts:
Sentence 0: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry.
  Fact 1.1: Bankman-Fried presented himself as the trustworthy face of a sometimes-shady industry.

Retrieved items:
<item 1>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 2>

<item 3>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
The trial of Sam Bankman-Fried is likely to be more consequential than just whether the man himself is found guilty. Depending on what evidence is introduced
during the trial, it could be rough for the entire crypto industry.

“How much damage can this trial do to the already beaten-down reputation of the industry at this point?” asks Yesha Yadav, a law professor at Vanderbilt
University. “This trial is going to be an excruciating moment for the industry because no one knows what kind of evidence might come out.”

Bankman-Fried, the founder of FTX and Alameda Research, is facing seven counts of criminal charges: two counts of wire fraud, and five counts of conspiracy
charges. FTX was a failed cryptocurrency exchange founded in 2019. According to a now-deleted profile from FTX investors Sequoia Capital, FTX was founded
because of Bankman-Fried’s frustration with other exchanges when he was running Alameda Research, his crypto trading firm.
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 5>

<item 6>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 6>

<item 7>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 10>
=== END ITEM id=a003-72eb50561fbeb609 ===

=== ITEM id=a016-aa40793277a448d6 ===
Gold sentences and facts:
Sentence 0: The Commanders are coming off a dud last week against the Giants, while Dallas is flying high.
  Fact 1.0: The Commanders are coming off a dud last week against the Giants.

Retrieved items:
<item 1>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
And all the effort is showing.

According to Joe Pompliano, “The NFL will bring in $20 billion in revenue this year, more than the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1
combined.” Coming to the strategy part, a curious mind would perhaps think can they take a leaf out of the NBA’s playbook and reinvigorate its appeal as it aims
to conquer international audiences?

Can the NFL take a leaf out of the NBA’s grisly past and glorious recovery?

The NFL is already the world’s richest league. Calling it an incredible feat would be a gross understatement. Consider this: the English Premier League, which
is broadcast in over 212 territories, has a revenue of just $6 billion, while football games are primarily viewed in America only. However, the NFL does not
just want to stop at $18 billion.
</item 1>

<item 2>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 2>

<item 3>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
However, the NFL does not just want to stop at $18 billion. It wants to become a $27 billion league in 4 years. But it must address a few skeletons in its
closet before reaching that goal.

If you were around in the 1970s (that would make you at least 50 years old today), you would know that the NBA did not have the mass appeal that it enjoys now.
With average game attendance hovering around 8,000 a game, it was one of the darkest periods of the basketball league. Drug abuse, brawls on the hardwood court
that sometimes spilled over onto coaches and players – it wasn’t a pretty sight. As reported in the Washington Post, even though there are no reliable figures
on cocaine use by players, “estimates by people in the game range from 40 percent to 75 percent.”
</item 3>

<item 4>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The October 1 matchup between the Chiefs and the Jets drew 27 million viewers. And became the most-watched Sunday TV show since the last Super Bowl! And the
last Super Bowl had an actual concert by Rihanna.

Revealed: The Dark Side Of Pat McAfee & Why Rob Gronkowski Is In The Firing Line

Kaepernick, who? Deflategate, what? In the media frenzy surrounding the two lovebirds, other issues get thrown out of the window. How can fans forget that the
NFL is also organizing games outside the American borders with a view to garnering international appeal? The NFL has scheduled 5 games abroad this season. It is
also addressing the concussion controversies. It recently added both new concussion rules and a new $100 million initiative called “Play Smart, Play Safe,” to
make the sport safer for players. And all the effort is showing.
</item 4>

<item 5>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 5>

<item 6>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
Hard to rank it any higher until we know what we’re dealing with.

Largest crowd ever for a game at Globe Life Field. #StraightUpTX pic.twitter.com/4I7NIB5eF8 — Chris Halicke (@ChrisHalicke) May 1, 2022

The old Rangers ballpark could drum up a Texas-sized atmosphere worthy of Stevie Ray Vaughan and Pat Green, but the new place — at least in the regular season —
feels very much like a multi-events space that happens to be hosting a ballgame. Maybe that changes in the playoffs? The Rangers haven’t been home since Sept.
24 when they won their last five games in Arlington, the last three of which went a long way toward keeping the Mariners out of the postseason.

With a closed roof containing all the noise, Globe Life is sure to get awfully loud, and nothing turns a glorified convention center into a real ballpark quite
like a late-inning rally in October.

No.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
But they also haven’t really beaten anyone. They were down 14 points to Carolina. Do you think they are coming back from 13 down against Philly? In my opinion,
no.”

Advertisement

• Dallas Cowboys: Is this team any better than previous Dallas teams that fell short?

What I think: It’s a tough time to evaluate the Cowboys fairly because they were so bad against San Francisco on a big stage in their most recent game. I just
don’t see where Dallas is better or going to get considerably better, especially on offense. And every time they enjoy success, they seem to enjoy it a little
too much. That is a reflection of their owner.

Exec comment: “Run defense and consistency on defense, and then offensively, what is their identity? They just are not very good. Their best games are when they
play great defense and get turnovers, but that is not going to happen enough, especially against the good teams.”

• Detroit Lions: Have they fixed their defense?
</item 7>

<item 8>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 8>

<item 9>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 9>

<item 10>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The hardcore football fans are pretty much done with the NFL harping about Travis Kelce and Taylor Swift. Reality TV star Rachel Lindsay, a longtime NFL fan,
recently slammed the “obnoxious” coverage, highlighting the “exhausting” uptick in Swifties obsessing over football. But that’s just one example of how fans are
finding this fixation a little too disrespectful towards the sport.

America’s Favorite Video Today

From the league’s humble beginnings in 1920 to the $18 billion juggernaut that it is today, the ride has been rocky with more than just a few small bumps along
the road. But with the ambitious target that NFL Commissioner, Roger Goodell, has set for 2027, is there a new strategy in the 104-year-old league’s master plan
to dominate the world sporting scene?

Why the ‘Traylor’ saga will be the key to Roger Goodell’s $27 billion target?
</item 10>
=== END ITEM id=a016-aa40793277a448d6 ===

=== ITEM id=a131-1bdd3436857a738d ===
Gold sentences and facts:
Sentence 0: Kalākaua (November 16, 1836 – January 20, 1891), born David Laʻ amea Kamananakapu Mahinulani Naloiaehuokalani Lumialani Kalākaua and sometimes called The Merrie Monarch, was the last king (and second to last monarch) of the Kingdom of Hawaiʻ i.
  Fact 2.3: Kalakaua is sometimes called The Merrie Monarch.

Retrieved items:
<item 1>
Arthur Tedder, 1st Baron Tedder
Marshal of the Royal Air Force Arthur William Tedder, 1st Baron Tedder, {'1': ", '2': ", '3': ", '4': "} (11 July 1890 – 3 June 1967) was a senior Royal Air
Force commander.

 He was a pilot and squadron commander in the Royal Flying Corps in the First World War and he went on to serve as a senior officer in the Royal Air Force
during the inter-war years when he served in Turkey, Great Britain and the Far East.

 During the Second World War, as Air Officer Commanding RAF Middle East Command, Tedder directed air operations in the Mediterranean and North Africa, including
the evacuation of Crete and "Operation Crusader" in North Africa.

 His bombing tactics became known as the "Tedder Carpet".

 Later in the war Tedder took command of Mediterranean Air Command and in that role was closely involved in the planning of the Allied invasion of Sicily and
then the Allied invasion of Italy.

 When Operation Overlord—the invasion of France—came to be planned, Tedder was appointed Deputy Supreme Commander at Supreme Headquarters Allied Expeditionary
Force under General Eisenhower.
</item 1>

<item 2>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 2>

<item 3>
Charles Hastings Judd
Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.

 He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
</item 3>

<item 4>
Edmund Mortimer, 5th Earl of March
Edmund de Mortimer, 5th Earl of March and 7th Earl of Ulster (6 November 1391 – 18 January 1425) was an English nobleman.

 A great-great-grandson of King Edward III of England, he was heir presumptive to King Richard II of England, his first cousin twice removed, when Richard II
was deposed in favour of Henry IV.

 Edmund Mortimer's claim to the throne was the basis of rebellions and plots against Henry IV and his son Henry V, and was later taken up by the House of York
in the Wars of the Roses, though Mortimer himself was an important and loyal vassal of Henry V and Henry VI.

 Edmund Mortimer was the last Earl of March of the Mortimer family.
</item 4>

<item 5>
Baron Plunket
Baron Plunket, of Newtown in the County of Cork, is a title in the Peerage of the United Kingdom.

 It was created in 1827 for the prominent Irish lawyer and Whig politician William Plunket.

 He served as Lord Chancellor of Ireland from 1830 and 1834 and again from 1835 to 1841.

 His eldest son, the second Baron, was Bishop of Tuam, Killala and Achonry between 1839 and 1866.

 He was succeeded by his younger brother, the third Baron.

 He was a barrister.

 His eldest son, the fourth Baron, served as Archbishop of Dublin between 1884 and 1897.

 He was succeeded by his eldest son, the fifth Baron.

 He was a diplomat and held office as Governor of New Zealand between 1904 and 1910.

 His grandson, Patrick, the seventh Baron, was Equerry to both King George VI and Queen Elizabeth II.

 He was succeeded by his younger brother, Robin who died in 2013 and was in turn succeeded by his nephew, Tyrone who was a Page of Honour to Queen Elizabeth II.
</item 5>

<item 6>
HMS Monmouth
Seven ships of the Royal Navy have been named HMS "Monmouth".

 Monmouth was the name of a castle and is now the name of a town in Wales; the name also recognises James Scott, 1st Duke of Monmouth, the "Black Duke".

 The first HMS "Monmouth" was named after the town but whilst the second one was being used the Monmouth Rebellion took place and this caused some concern to
the crew.

 Since that time, all of "Monmouth"s have had their names painted in black and the name and the crest are covered because of the shame of the Duke of Monmouth's
execution.

 Today the officers of the latest HMS "Monmouth" carry black handkerchiefs and the ship is known as the Black Duke.

 However the town of Monmouth still gives the freedom of the town to the ship's crew and talks of the "enduring link" between the town and the ship.
</item 6>

<item 7>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 7>

<item 8>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 8>

<item 9>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 9>

<item 10>
Corps of Military Staff Clerks
The Corps of Military Staff Clerks was an administrative corps of the Canadian Army.

 The Headquarters Corps of Military Staff Clerks was established in 1905.
</item 10>
=== END ITEM id=a131-1bdd3436857a738d ===

=== ITEM id=a043-7f9ecbfa97a507a0 ===
Gold sentences and facts:
Sentence 0: It would be Harbaugh's second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh lied to investigators.
  Fact 0.2: The self-imposed ban was amid an NCAA investigation.

Retrieved items:
<item 1>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Rice on Saturday and at South Florida on Nov. 25.

I am extending an open invitation to Jim Harbaugh to join me on the sidelines for the next two Charlotte football games. It would be my honor to stand next to
you again as we did for three years — Biff poggi (@BiffPoggi) November 17, 2023

Thursday, Nov. 16

Michigan accepts Harbaugh's three-game suspension

Michigan has dropped its litigation against the Big Ten over Harbaugh's three-game suspension, and it will accept the punishment levied against the head coach.

From Michigan:

"This morning, the University, Coach Harbaugh, and the Big Ten resolved their pending litigation. The Conference agreed to close its investigation, and the
University and Coach Harbaugh agreed to accept the three-game suspension. Coach Harbaugh, with the University's support, decided to accept this sanction to
return the focus to our student-athletes and their performance on the field.
</item 1>

<item 2>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.



"Michigan has been found in violation of the Big Ten Sportsmanship Policy for conducting an impermissible, in-person scouting operation over multiple years..."
https://t.co/htnUkshmnF — Chris Vannini (@ChrisVannini) November 10, 2023

Jim Harbaugh joins Michigan for trip to State College

According to the sleuths on the ground in Ann Arbor, Harbaugh is on the team bus and heading to the airport with his team as Michigan awaits word on a potential
suspension.

Jim Harbaugh boarding the bus pic.twitter.com/CWBJbwEEhi — Austin Meek (@byAustinMeek) November 10, 2023

Michigan still awaiting word from Big Ten

According to ESPN's Adam Rittenberg, Michigan still hasn't heard about an impending punishment from the Big Ten as its 1 p.m. ET flight looms.
</item 2>

<item 3>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Nicole Auerbach (@NicoleAuerbach) November 11, 2023

Harbaugh hearing coming down to wire before Penn State game

ESPN's Pete Thamel reported it's unclear Harbaugh will be granted a hearing in time to be able to coach against Penn State. With kickoff scheduled for noon ET,
Thamel said "the drama's going to go right up 'till noon."

Latest from ⁦@CollegeGameDay⁩ on Jim Harbaugh’s TRO legal filing. Will he coach Michigan at Penn State today? pic.twitter.com/bz3R1eYxI9 — Pete Thamel
(@PeteThamel) November 11, 2023

The case is reportedly in the hands of Judge Carol Kunhke, a Michigan alumnus.

Harbaugh was not on the Michigan team bus with other coaches and players when it arrived at Beaver Stadium Saturday morning.

Michigan players and coaches arrive. No Jim Harbaugh on the bus.
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
We have to wait for the news, but a suspension is certainly on the table. — Heather Dinich (@CFBHeather) November 10, 2023

If it’s more than 2 games or $10k, the Big Ten will need special permission from an executive board composed of league reps https://t.co/6Z7jb2zqH4 — Heather
Dinich (@CFBHeather) November 10, 2023

Punishment expected on Friday

The college football world will wait with bated breath on Friday as it awaits what Michigan and Jim Harbaugh will face from the Big Ten. ESPN's Heather Dinich
reported the punishment is expected to be announced early Friday afternoon. She noted the team is leaving for State College at 1 p.m. CT for the Penn State game
on Saturday.

sources indicate a Big Ten response is coming early this afternoon.
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Here is Michigan’s 10-page letter to the Big Ten responding to the sportsmanship investigation, reporting with @RossDellenger 1/3 pic.twitter.com/P0oYgIbS67 —
Dan Wetzel (@DanWetzel) November 9, 2023

No timetable for Big Ten decision

The Big Ten is reviewing the "lengthy" official response from Michigan and Jim Harbaugh and it's unclear when a potential ruling might come down, says ESPN's
Pete Thamel. His colleague Adam Rittenberg reported Wednesday that a decision was expected by Thursday.

Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh. There’s no indication of any timetable on a decision on
potential Big Ten discipline in the case of Michigan’s alleged illegal signal stealing.
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Within it, the school warned Big Ten commissioner Tony Pettiti against levying "premature" disciplinary action against Jim Harbaugh and alleged that in-person
scouting and sign-stealing are commonplace across the NCAA.

"The conference should act cautiously when setting precedent given the reality that in-person scouting, collusion among opponents, and other questionable
practices may well be far more prevalent than believed," the letter read.

Michigan also included with the letter "two attached documents of Michigan's offensive and defensive signs allegedly stolen and shared by other Big Ten teams,"
per Yahoo Sports.

Conference awaits Michigan response

The Big Ten expects a formal response from Michigan to the notice of discipline by the end of the day Wednesday, ESPN's Pete Thamel reports. His colleague Adam
Rittenberg adds that a ruling is expected to come down Thursday.

Sources: The Big Ten expects a response from Michigan on the notice of discipline by the end of day on Wednesday. Don’t expect any ruling by the Big Ten on
Wednesday, as they are expected to take time to absorb Michigan’s response.
</item 6>

<item 7>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Although the Big Ten has closed its investigation, we are continuing to cooperate with the NCAA as it moves forward with its ongoing investigation. Consistent
with our commitment to integrity, we will continue to take the appropriate actions, including disciplinary measures, based on information we obtain. Earlier
today, Michigan Athletics relieved Chris Partridge of his duties as a member of the Michigan Football staff. Due to employee privacy laws, we are unable to
comment further."

Because Michigan and Harbaugh on Thursday dropped their litigation against the Big Ten, he will not be on Michigan's sideline this week against Maryland and
next week against Ohio State to end the regular season.

Former Wolverines assistant and Harbaugh confidant Biff Poggi on Friday offered his former boss an invitation. Poggi, in his first year as Charlotte's head
coach, asked Harbaugh to join the 49ers for their games vs. Rice on Saturday and at South Florida on Nov. 25.
</item 7>

<item 8>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
ET flight looms.

Michigan's administration has not heard from the Big Ten as of this hour, per sources. The team will begin its preparations to leave for Penn State around 1
p.m. ET. Really going down to the wire with any discipline, which, as several sources noted, isn't really fair to players. — Adam Rittenberg (@ESPNRittenberg)
November 10, 2023

Harbaugh still expected to fly to State College

Dinich reported earlier on Friday that punishment to Michigan could come around the time the team is expected to get on a plane and fly to Pennsylvania for its
game against Penn State. But regardless of when the punishment is announced, Harbaugh's attorney, Tom Mars, reportedly told ESPN the coach will be headed to
State College.

"I'm confident he's going to be on that plane one way or another," Mars reportedly told ESPN.
</item 8>

<item 9>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
On the field, Michigan is in the midst of what could be a historic season. Off the field, the the focus has rarely been on the team's success.

An investigation into Michigan's alleged sign-stealing scheme, centered on now-fired staffer Connor Stalions, continues to evolve by the hour, with one update
after another adding another wrinkle to the saga that began in October. The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a
plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy
Valley.

The Sporting News is tracking the latest updates from the Michigan sign-stealing scandal. Follow along with the newest developments below.

MORE ON MICHIGAN SIGN-STEALING:

Michigan sign-stealing scandal updates

Friday, Nov.
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”
</item 10>
=== END ITEM id=a043-7f9ecbfa97a507a0 ===

=== ITEM id=a060-8dbe62248ad588d8 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 1.2: The condition was that the amount withdrawn was less than FTX's total trading revenue.

Retrieved items:
<item 1>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Bankman-Fried faces seven federal charges, including wire fraud, securities fraud and money laundering, that could put him in prison for the rest of his life.
Thus far, Bankman-Fried, 31, has remained mostly quiet in court intently listening to witnesses and at times writing notes to his attorneys. But as Wang
testified against him, Bankman-Fried looked visibly upset, shifting his gaze from his former friend to the ground, and at one point putting his head in his
hands.

Sam Bankman-Fried listens as Assistant U.S. Attorney Nicolas Roos questions Gary Wang during Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Wang, 30, was technology chief for FTX, which spiraled into bankruptcy in November.
</item 1>

<item 2>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
How much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.

Shortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried
were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in
finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.

When Yedidia first took the stand, Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his
testimony. Why did he make such a deal with the government, she asked.

“I was concerned that I had unwittingly written code that contributed to a crime,” he said.
</item 2>

<item 3>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
It is a thankless task to cross-examine a customer whose money is gone

In November 2022, things went bad for Julliard. He followed Bankman-Fried on Twitter, and read aloud the “FTX is fine. Assets are fine” tweets, along with “FTX
has enough to cover all client holdings. We don’t invest client assets” and a few others, which gave Julliard the impression that his money was there — the
problem might have been technical (anti-spam measures) or regulatory. When he tried to get his money out on November 8th, it was too late. We saw screenshots of
his withdrawal attempts: $20,000 USD and about 4 Bitcoin, which were worth about $20,000 at the time: about $100,000 money, inaccessible.

It is a thankless task to cross-examine a customer whose money is gone, but Cohen tried anyway.
</item 3>

<item 4>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 4>

<item 5>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Sure, Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of
the people who are closest to him.

Of course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm Alameda Research purloining its
customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that they should have had access to at
any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life savings lost.

There is a simultaneous drama of ruined lives playing out in the courtroom

Bankman-Fried is charged with seven counts: two counts of wire fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been
struck by how bizarre it all is.
</item 5>

<item 6>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.
</item 6>

<item 7>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
He did. But — I did hear these words uttered aloud in a court of law this morning, I am not creative enough to make this kind of thing up — his employees told
him “they were busy and I should stop asking questions because it was distracting.”

Yedidia — Bankman-Fried’s college friend, Bahamas roommate, and employee at FTX — had testified that he’d asked Bankman-Fried about the $8 billion hole on a
padel tennis court in their luxury complex in June or July. Today, in testimony, Bankman-Fried seemed to be trying to deny that conversation had ever taken
place. It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that
Bankman-Fried admitted he’d been told.
</item 7>

<item 8>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 9>

<item 10>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 10>
=== END ITEM id=a060-8dbe62248ad588d8 ===

=== ITEM id=a122-85cc761972810b23 ===
Gold sentences and facts:
Sentence 0: The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and artificiality.
  Fact 2.3: The Decadent Movement followed an aesthetic ideology of excess.

Retrieved items:
<item 1>
Guillaume Morissette
Guillaume Morissette (born 1984) is a Canadian fiction writer and poet based in Montreal, Quebec.

 His work has frequently been associated with the Alt Lit movement, with Dazed & Confused magazine describing him as "Canada's Alt Lit poster boy."

 He has published stories, poems and essays online and in print, in venues such as "Maisonneuve", "Little Brother", "Broken Pencil", "Shabby Doll House" and
"Thought Catalog", and was listed as one of CBC Books' "Writers to Watch" for 2014.
</item 1>

<item 2>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 2>

<item 3>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 3>

<item 4>
Decadent movement
The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and
artificiality.

 The visual artist Félicien Rops's body of work and Joris-Karl Huysmans's novel "Against Nature" (1884) are considered the prime examples of the decadent
movement.

 It first flourished in France and then spread throughout Europe and to the United States.

 The movement was characterized by self-disgust, sickness at the world, general skepticism, delight in perversion and employment of crude humor and a belief in
the superiority of human creativity over logic and the natural world.
</item 4>

<item 5>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 5>

<item 6>
Hermann Brunn
Karl Hermann Brunn (1 August 1862 – 20 September 1939) was a German mathematician, known for his work in convex geometry (see Brunn–Minkowski inequality) and in
knot theory.

 Brunnian links are named after him, as his 1892 article "Über Verkettung" included examples of such links.
</item 6>

<item 7>
Cult MTL
Cult MTL is an English language arts, culture and news website and monthly print publication, based in Montreal, Quebec, Canada.

 Its first print edition appeared on 7 September 2012.

 It was created only a few months after Montreal's last English-language alternative weekly, "Montreal Mirror", was unceremoniously closed by its parent
company, Quebecor.

 The founding editor's of "Cult MTL" were also involved with the "Mirror".

 In August 2013 the print version of the magazine was started.
</item 7>

<item 8>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 8>

<item 9>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 9>

<item 10>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 10>
=== END ITEM id=a122-85cc761972810b23 ===

=== ITEM id=a049-c6e13c59a01ff032 ===
Gold sentences and facts:
Sentence 0: Second-last of ten on the table after a fourth abject loss in barely three weeks, this time by eight wickets and nearly 25 overs to spare against Sri Lanka having been bowled out for a miserable 156 in Bengaluru, the 2019 champions are now only mathematically a chance of scraping through to the semi-finals.
  Fact 1.4: The loss was against Sri Lanka.

Retrieved items:
<item 1>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India moved to finals and beat Sri Lanka to win their second World Cup title.



2015 World Cup, against Australia

India were playing against Australia in the second semi-final of the 2015 World Cup in Sydney on March 26. Australia outplayed India in almost all departments
of the game.

Australian Man-of-the-Match Steve Smith (105 runs in 93 balls) and Aaron Finch (81 runs in 116 balls) were the guiding forces in taking Australia to 328 runs
for 7 wickets in 50 overs.

In reply, India needed a few big innings, but apart from skipper MS Dhoni (65 runs in 65 balls), Shikhar Dhawan (45 runs in 34 balls), and Ajinkya Rahane (44
runs in 68 balls), the batting order collapsed miserably as India lost the match by 95 runs.



2019 World Cup, against New Zealand

The cricket World Cup returned to England for the fifth time.
</item 1>

<item 2>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2003 World Cup, against Kenya

India were playing underdogs Kenya in the last-four stage match in Durban on March 20.

With Man-of-the-Match India skipper Sourav Ganguly scoring an unbeaten 111 runs in 114 balls and Man-of-the-Tournament Tendulkar scoring an 111-ball 83, India
comfortably scored 270 for four wickets in 50 overs.

In reply, Kenya were no match for Indian dominance, and barring their skipper Steve Tikolo, who top-scored with 56 runs, no Kenyan batsman could score more than
29 runs.

With Zaheer Khan taking 3 wickets in 9.2 overs for just 14 runs, India could easily beat Kenya by 91 runs.

Though India breezed into semi-finals easily, the tournament is also remembered for Kenya's best run in the World Cup as the team from a country with limited
cricket facilities and no proper grassroots structure could reach as far as semi-finals.
</item 2>

<item 3>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India were playing their sixth semi-finals at a place where they played their first World Cup semi-final in 1983, Manchester. However, with a different team-
New Zealand.

Indian bowlers raised hopes when they restricted New Zealand to 239 for 8 in 50 overs with Bhuvneshwar Kumar taking three wickets.

Ross Taylor 74 runs (90 balls) and skipper Kane Willamson 67 runs (95 balls) were the top scorers for New Zealand. But Kiwi Man-of-the-Match bowler Matt Henry
sent openers KL Rahul and Rohit Sharma back to the pavilion soon, followed by the dismissal of Dinesh Karthik in the 10th over of India's innings.

With the Indian top-order failing, Ravindra Jadeja (77 runs in 59 balls) and wicketkeeper MS Dhoni (50 runs in 72 balls) tried to steady the Indian ship.

But once they fell, it was all over for India as they lost the battle by 18 runs.
</item 3>

<item 4>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
In reply, Mohinder Amarnath played a useful innings of 46, while Yashpal Sharma scored a fine 61.

Sandeep Patil played a fiery innings of 51 not out in just 32 balls as India stormed into semifinals for the first time, beating England by six wickets with 5.2
overs to spare.

Also Read: ICC World Cup 2023: List of centuries; Most tons in World Cup so far

1987 Reliance Cup, against England

Four years later, the World Cup was being played outside England for the first time, and it was India which was hosting it.

In a co-incidence, India was against England in the semi-finals once again, at the Wankhede Stadium, Mumbai, on November 5.

This time, England's Man-of-the-Match skipper Graham Gooch smashed 115 runs in 136 balls to lay a solid foundation for his team's fighting total of 254 for 6
wickets.
</item 4>

<item 5>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India, one of the favourites to win the World Cup, were very much on the path to victory till Mohammad Azharuddin, who top-scored with 64 runs was at the
crease.

But with the fall of his wicket, the Indian lower order collapsed like a house of placards as India lost the match by 35 runs.

1996 Wills World Cup, against Sri Lanka

The 1996 Cricket World Cup is known for Sri Lanka's emergence as a cricketing power on the horizon of world cricket and the beginning of Sachin Tendulkar's
extraordinary dominance in batting history.

Playing in the first semi-final at the Eden Gardens in Kolkata on March 13, India were favourites again, and they raised their hopes when they managed to
restrict Sri Lanka to 251 for 8 in 50 overs despite Man-of-the-Match Aravinda de Silva's 66 and Roshan Mahanama's 58.
</item 5>

<item 6>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
ICC World Cup 2023: Rohit Sharma-led India are playing Netherlands in their ninth ICC World Cup 2023 round-robin match at the M Chinnaswamy Stadium in Bengaluru
on Sunday (November 12, 2023). India have already booked a place in semi-finals, thrashing Sri Lanka by 302 runs at the Wankhede Stadium in Mumbai on November
2, 2023. India will face New Zealand in the ongoing World Cup semi-final match at the Wankhede Stadium in Mumbai on Wednesday (November 15). This is the seventh
time India have entered into the last-four stage of the 50-over World Cup.

Also Read: ICC Cricket World Cup 2023 Prize Money: How much finalists, semi-finalists, and other teams will win at ODI World Cup

India first entered semifinals of the World Cup at the 1983 Prudential Cup in England, where they emerged as the tournament winners.
</item 6>

<item 7>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2011 World Cup, against Pakistan

Fans from India and Pakistan wouldn't have expected a better occasion than a World Cup for the meeting of two teams.

The World Cup returned to India after 1996, and India and Pakistan were facing each other in the second semi-final at the IS Bindra Stadium in Mohali on March
30.

Man-of-the-Match Tendulkar rose to the occasion, scoring 85 runs for India.

Virender Sehwag scored 38 while Suresh Raina remained unbeaten on 36 as India scored 260 for 9 in 50 overs.

Pakistan seamer Wahab Riaz took 5 for 46 in 10 overs. In reply, Pakistan batsmen Misbah-ul-Haq (56 runs) and Mohammad Hafeez (43 runs) showed some resistance
from Pakistan, but in lack of a big innings lower down the order hit Pakistan's prospects badly as they were all out for 231 runs in 49.5 overs.
</item 7>

<item 8>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Alyssa Healy has rejected any suggestion Australia’s stranglehold on international women’s cricket is slipping, saying their Test loss to India is barely a blip
on the team’s record.

Australia stumbled to their first Test defeat to India by eight wickets in Mumbai on Sunday, after being outplayed by the hosts in foreign spinning conditions.

The loss comes after one of the more difficult periods for the Australian side in the past six years, with the retirements of Meg Lanning and Rachael Haynes.

Facebook Twitter Whatsapp Reddit Email Share

After winning their fourth straight global ICC event earlier this year in the 50-over World Cup, Australia retained the Ashes via a drawn multi-format series in
England in July.

That tour included losing four of six white-ball matches, with Australia’s win in the Test enough to have them level on points with England and hold onto the
trophy.
</item 8>

<item 9>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
Also Read: IND vs NZ FREE Live Streaming: When and How to watch India vs New Zealand Cricket World Cup 2023 Semi-Final Match Live on Web, TV, mobile apps online

Also Read: AUS vs SA FREE Live Streaming: When and How to watch Australia vs South Africa Cricket World Cup 2023 Semi-Final Match Live on Web, TV, mobile apps
online

1983 Prudential Cup, against England

Having lost in the first round of the 1975 and 1979 World Cups, India were underdogs in the third World Cup, which was being held in England for the third
consecutive time.

However, in the World Cup, India wronged many of their critics, reaching semi-finals and playing against mighty hosts England in Manchester on June 22.

Skipper Kapil Dev struck three wickets while Roger Binny and Man-of-the-Match Mohinder Amarnath took two each as India restricted England to 213 all out in 60
overs.
</item 9>

<item 10>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
They moved into the last-four stage on the next two occasions when the World Cup was held in India, in 1987 and 1996.

India were in semi-finals in 2003 in South Africa.

Eight years later, they were again in last four at the 2011 World Cup as hosts, and repeated their feat in 2015, 2019, and 2023.

It means that in their last six World Cups, India have been in the semi-finals on five occasions.

Also Read: ICC Cricket World Cup 2023 semi-final schedule: India to play New Zealand in Mumbai; South Africa to face Australia in Kolkata; know venue, squads

In this write-up, ZeeBiz will take you through the memory lane of India's semifinal journey in the biggest cricket extravaganza in the world.
</item 10>
=== END ITEM id=a049-c6e13c59a01ff032 ===

=== ITEM id=a133-341e5bd338da21a0 ===
Gold sentences and facts:
Sentence 0:  A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work behind the camera, with Griffith and Joseph Henabery as his mentors.
  Fact 2.6: Joseph Henabery was a mentor of Ridley Scott.

Retrieved items:
<item 1>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 1>

<item 2>
Enrico Cocozza
Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.

 His often surreal films were mainly filmed in and around the town of Wishaw in Scotland, where his family owned the popular Belhaven Cafe.

 These include "Chick's Day" (1950), a prize winner at the 1951 Scottish Amateur Film Festival, "The Living Ghost" (1957), and "Glasgow's Docklands" (1959).

 Illness later forced him to give up making films and he spent most of his working life teaching at the University of Strathclyde in Glasgow.

 In 2001 he was the subject of the Channel 4 documentary "Artery: the Story of Enrico Cocozza".
</item 2>

<item 3>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 3>

<item 4>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 4>

<item 5>
Graph query row:
person: Ridley Scott
connected_works: ['The Polar Bears', 'Alien', 'Blade Runner', 'Gladiator', 'The Martian']
output_count: 5
compared_person: Elmer Clifton
compared_works: ['The Flame of Youth', 'Seven Doors to Death', 'Gangsters of the Frontier']
compared_output_count: 3
comparison: Ridley Scott has more
</item 5>

<item 6>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 6>

<item 7>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 7>

<item 8>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 8>

<item 9>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 9>

<item 10>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 10>
=== END ITEM id=a133-341e5bd338da21a0 ===

=== ITEM id=a000-35ccf7b1a23ac1aa ===
Gold sentences and facts:
Sentence 0: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The theory is based on the board's language.

Retrieved items:
<item 1>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.
</item 1>

<item 2>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman also co-founded Worldcoin, a biometric and cryptocurrency project that’s been scanning people’s eyeballs with the goal of creating a vast digital
identity and financial network.

___

Associated Press business writers Haleluya Hadero in New York, Kelvin Chan in London and Michael Liedtke and David Hamilton in San Francisco contributed to this
report.
</item 2>

<item 3>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
—

November 18

“Not … in response to malfeasance”

In an internal memo obtained by Axios sent Saturday morning, OpenAI COO Brad Lightcap said yesterday’s announcement “took [the management team] by surprise” and
that management had had “multiple conversations with the board to try to better understand the reasons and process behind their decision.” Discussions were
ongoing as of Saturday morning, per the memo.

“We can say definitively that the board’s decision was not made in response to malfeasance or anything related to our financial, business, safety, or
security/privacy practices,” Lightcap added. “This was a breakdown in communication between Sam and the board … We still share your concerns about how the
process has been handled, are working to resolve the situation, and will provide updates as we’re able.”

OpenAI’s funding in jeopardy

The planned sale of OpenAI employee shares that would value the startup at about $86 billion could be in jeopardy.
</item 3>

<item 4>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
After sending a Google Meet link, Brockman was told that he was being removed from the board as chairman “but was vital to the company and would retain his
role” as president, and that Altman had been fired.

Altman’s firing publicly announced

OpenAI published a post on its blog announcing the executive shake-up. The company’s management team was aware shortly after.

i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved working with such talented people.
will have more to say about what’s next later. 🫡 — Sam Altman (@sama) November 17, 2023

All-hands meeting

OpenAI held an all-hands meeting Friday afternoon during which Sutskever defended Altman’s ouster. He dismissed suggestions that pushing Altman out amounted to
a “hostile takeover,” and claimed that it was necessary to protect OpenAI’s mission of “making AI beneficial to humanity.”
</item 4>

<item 5>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He said OpenAI still has a “deep bench of technical leaders” but its next executives will have to steer it through the challenges of scaling the business and
meeting the expectations of regulators and society.

Forrester analyst Rowan Curran speculated that Altman’s departure, “while sudden,” did not likely reflect deeper business problems.

“This seems to be a case of an executive transition that was about issues with the individual in question, and not with the underlying technology or business,”
Curran said.

Altman has a number of possible next steps. Even while running OpenAI, he placed large bets on several other ambitious projects.

Among them are Helion Energy, for developing fusion reactors that could produce prodigious amounts of energy from the hydrogen in seawater, and Retro
Biosciences, which aims to add 10 years to the human lifespan using biotechnology.
</item 5>

<item 6>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 6>

<item 7>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
one takeaway: go tell your friends how great you think they are. — Sam Altman (@sama) November 18, 2023

Investors pushing for Altman’s return

Investors — furious at the turn of events — are reportedly exerting pressure on OpenAI’s board to reinstate Altman, going so far as to recruit Microsoft.
Nadella is said to be sympathetic.

Board agrees to reverse course — in principle

The Verge reports that the board agreed in principle to resign and to allow Altman and Brockman to return. It waffled, however, missing a deadline yesterday by
which many OpenAI staffers were set to leave the company. Altman is said to be ambivalent about coming back and asking for “significant” governance changes.

—

November 17

Brockman demoted

Brockman says he got a text from Sutskever shortly after noon on Friday asking for a quick call.
</item 7>

<item 8>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Later Monday, that number climbed to over 650.

Altman and Brockman considering return

As reported by The Verge, Altman’s move to Microsoft isn’t a done deal — and both Altman and Brockman are still open to returning to OpenAI. That is, if the
remaining board members who initially fired him step aside.

OpenAI board considers merger

OpenAI’s board of directors approached Dario Amodei, the co-founder and CEO of rival large-language model developer Anthropic, about a potential merger of the
two companies, The Information reports. The approach was part of an effort by OpenAI to persuade Amodei to replace Altman as CEO — but Amodei quickly turned
down the CEO offer.

—

November 19

Altman to meet at OpenAI HQ

According to The Information, Altman is expected to meet at OpenAI’s San Francisco headquarters as executives at OpenAI push to have him reinstated as CEO.
</item 8>

<item 9>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 9>

<item 10>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.
</item 10>
=== END ITEM id=a000-35ccf7b1a23ac1aa ===

=== ITEM id=a115-c0374516833d00c8 ===
Gold sentences and facts:
Sentence 0: Lyle Martin Alzado (April 3, 1949 – May 14, 1992) was a professional American football defensive end of the National Football League famous for his intense and intimidating style of play.
  Fact 2.2: Lyle Martin Alzado was a professional American football defensive end.

Retrieved items:
<item 1>
Alec Gibson
Alec Raymond Gibson (born December 9, 1963 in Columbus, Ohio) is a former American football defensive end in the National Football League for the Washington
Redskins.

 Gibson played in three games for the Washington Redskins in 1987.

 The Washington Redskins went 3-0 during the strike.

 Defeating the St. Louis Cardinals, New York Giants, and Dallas Cowboys.

 Gibson caused Hall of Famer Tony Dorsett to fumble the football twice on Monday Night Football in Dallas the last replacement game played in 1987.

 He played college football at Ventura Junior College where he was an All-American defensive tackle and at the University of Illinois as a defensive end.

 In 1988, he played Arena football for the New York Knights.

 Gibson played 11 of 12 games before injuring his right knee (ACL) which put an end to his career.
</item 1>

<item 2>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 2>

<item 3>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 3>

<item 4>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 4>

<item 5>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 5>

<item 6>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 6>

<item 7>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 7>

<item 8>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 8>

<item 9>
Batman (1989 film)
Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.

 It is the first installment of Warner Bros.' initial "Batman" film series.

 The film stars Michael Keaton as Bruce Wayne/Batman and Jack Nicholson as The Joker, alongside Kim Basinger, Robert Wuhl, Pat Hingle, Billy Dee Williams,
Michael Gough and Jack Palance.

 The film takes place early in the title character's war on crime, and depicts a battle with his arch-nemesis the Joker.
</item 9>

<item 10>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 10>
=== END ITEM id=a115-c0374516833d00c8 ===

=== ITEM id=a079-5a6e13414d4d6d7a ===
Gold sentences and facts:
Sentence 0:  It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
  Fact 2.3: TD Garden has been known as FleetCenter.

Retrieved items:
<item 1>
Sports Legends Museum at Camden Yards
Sports Legends Museum at Camden Yards was a non-profit sports museum in Baltimore, Maryland, United States, owned and operated by the Babe Ruth Birthplace &
Museum.

 It opened on May 14, 2005, with the daughter of celebrated baseball player Babe Ruth in attendance.

 After ten years of operation the museum closed abruptly on October 12, 2015 after failing to reach an agreement with the Maryland Stadium Authority for the
continued use of Camden Station.

 The 22000 sqft museum was adjacent to the main gate of Oriole Park at Camden Yards and has artifacts and interactive exhibits profiling Maryland’s sports
history.

 Exhibits included such area teams as the Baltimore Orioles, Baltimore Ravens, Baltimore Colts, Maryland Terrapins, Baltimore Elite Giants, Baltimore Black Sox,
and the Baltimore Blast.
</item 1>

<item 2>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 2>

<item 3>
Sports Museum of America
The Sports Museum of America (SmA) was the United States' first national sports museum dedicated to the history and cultural significance of sports in America.

 It opened in May 2008 and closed less than nine months later, in February 2009.
</item 3>

<item 4>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 4>

<item 5>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 5>

<item 6>
Papa Murphy's Park
Papa Murphy's Park (formerly known as the Cal Expo Multi-Use Sports Field Facility and Bonney Field) is a sports venue located on the grounds of Cal Expo in
Sacramento, California.

 The soccer-specific stadium has a capacity of 11,569 and includes a full-sized (120 x 80 yard) soccer field.

 Papa Murphy's Park is the current home of Sacramento Republic FC soccer team and former home of PRO Rugby team, Sacramento Express.
</item 6>

<item 7>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 7>

<item 8>
Estonian Sports Museum
The Estonian Sports and Olympic Museum, founded in 1963 and modernised in 2001, is the largest sports museum in the Baltic states.

 The museum is located on Rüütli street in Tartu, Estonia and the museum has a small outpost in Otepää that concentrates on the Winter sports.

 Before 2016, Estonian Sports and Olympic Museum was named Estonian Sports Museum.
</item 8>

<item 9>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 9>

<item 10>
Fanattic Sports Museum
Fanattic Sports Museum (FSM) is a sports museum, dedicated to various types of sports and located in Ecospace Business Park, New Town, Kolkata, India.

 Established in January 2017, FSM is known as the first sports museum in Kolkata as well as in West Bengal.
</item 10>
=== END ITEM id=a079-5a6e13414d4d6d7a ===

=== ITEM id=a025-e260d70b2e7ef22f ===
Gold sentences and facts:
Sentence 0: The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five catches, 157 yards, and two TDs.
  Fact 0.6: Tyreek Hill had two TDs against the Commanders.

Retrieved items:
<item 1>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.

Fantasy WR PPR Rankings Week 13: Who to start, sit at wide receiver

Rankings based on PPR scoring formats

Rank Player

1 Tyreek Hill, Dolphins @ Commanders

2 CeeDee Lamb, Cowboys vs. Seahawks

3 Amon-Ra St. Brown, Lions @ Saints

4 Keenan Allen, Chargers @ Patriots

5 Mike Evans, Buccaneers vs. Panthers

6 A.J. Brown, Eagles vs. 49ers

7 Michael Pittman Jr., Colts @ Titans

8 Ja'Marr Chase, Bengals @ Jaguars

9 Jaylen Waddle, Dolphins @ Commanders

10 Tank Dell, Texans vs. Broncos

11 Brandon Aiyuk, 49ers @ Eagles

12 Garrett Wilson, Jets vs. Falcons

13 Calvin Ridley, Jaguars vs.
</item 1>

<item 2>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 2>

<item 3>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 3>

<item 4>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 4>

<item 5>
Despite a 13.5-point spread, the Dolphins were the only team among 223 cited to both win and cover; six of 17 underdogs won. Tyreek Hill caught six passes for
163 yards and a touchdown against Carolina, reaching 812 receiving yards through six games—second in NFL history behind Don Hutson’s 819 in 1942. The article
also promoted *The Football 100*.
</item 5>

<item 6>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
There will be many difficult decisions to be made in Week 16. It is the Fantasy Football semifinals in most leagues and there is a certain finality to every
decision. There's no shortage of things to stress over as a Fantasy manager. Whether or not you start Rashee Rice shouldn't be one of them. While Rice is just
WR29 on the season, that's mostly due to a slow start, and his hot start has pushed him into must-start, borderline top-12 territory.

The Chiefs' star rookie has four straight games with at least nine targets and he's scored at least 14 PPR Fantasy points in all of those games. Since Week 12,
only Deebo Samuel and CeeDee Lamb have scored more PPR Fantasy points that Rice.

On paper, the matchup doesn't look great, because the Raiders have been above average in terms of limiting the Fantasy production of wide receivers.
</item 6>

<item 7>
Week 13’s six-team bye (Ravens, Bills, Vikings, Raiders, Giants, Bears) and injuries thinned WR options. Top PPR picks: Tyreek Hill, CeeDee Lamb, Amon-Ra St.
Brown, Keenan Allen and Mike Evans; sleepers included Curtis Samuel, Jayden Reed, Lynn Bowden Jr., Jonathan Mingo and Cedric Tillman. Week 14 had only
Commanders/Cardinals on bye. Hill again led rankings,
</item 7>

<item 8>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Which WRs are on bye in Week 14?

The final set of bye weeks sees only the Commanders and Cardinals idle, resulting in the losses of Terry McLaurin, Marquise Brown, Jahan Dotson, Curtis Samuel,
Rondale Moore, and Greg Dortch.

WEEK 14 PPR RANKINGS:

QBs | RBs | TEs | D/STs | Kickers

Even worse, Washington and Arizona sit as two of the weakest pass defenses in the league, so fantasy owners won't be able to count on big stats from their
receivers going against the Commanders and Cardinals.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy WRs in Week 14?

The league's leading receiver, Tyreek Hill (vs.
</item 8>

<item 9>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 9>

<item 10>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Fantasy WR PPR Rankings Week 14: Who to start, sit at wide receiver

Rankings based on full-point PPR scoring formats

Rank Player

1 Tyreek Hill, Dolphins vs. Titans

2 CeeDee Lamb, Cowboys vs. Eagles

3 Amon-Ra St. Brown, Lions @ Bears

4 A.J. Brown, Eagles @ Cowboys

5 Keenan Allen, Chargers vs. Broncos

6 Justin Jefferson, Vikings @ Raiders

7 Stefon Diggs, Bills @ Chiefs

8 Mike Evans, Buccaneers @ Falcons

9 Michael Pittman Jr., Colts @ Bengals

10 Davante Adams, Raiders vs. Vikings

11 Brandon Aiyuk, 49ers vs. Seahawks

12 Nico Collins, Texans @ Jets

13 Ja'Marr Chase, Bengals vs. Colts

14 Cooper Kupp, Rams @ Ravens

15 Chris Olave, Saints vs.
</item 10>
=== END ITEM id=a025-e260d70b2e7ef22f ===
