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

=== ITEM id=99176d6932fabd62 ===
Gold sentences and facts:
Sentence 0: Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.
  Fact 0.0: Kenneth William Kwapis, also known as Ken Kwapis, was born on August 17, 1957.
  Fact 0.1: Ken Kwapis is an American film director.
  Fact 0.2: Ken Kwapis is an American television director.
  Fact 0.3: Ken Kwapis is an American screenwriter.
Sentence 1:  He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
  Fact 1.0: He specialized in the single-camera sitcom in the 1990s.
  Fact 1.1: He specialized in the single-camera sitcom in the 2000s.
  Fact 1.2: He has directed feature films.
  Fact 1.3: He directed Sesame Street Presents Follow That Bird, released in 1985.
  Fact 1.4: He directed The Sisterhood of the Traveling Pants, released in 2005.
  Fact 1.5: He directed He's Just Not That into You, released in 2009.
Sentence 2:  As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).
  Fact 2.0: As a filmmaker, he has directed Valle de Lágrimas, released in 2006.
  Fact 2.1: As a filmmaker, he has directed Peace Grows, released in 2004.

Retrieved items:
<item 1>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 1>

<item 2>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 2>

<item 3>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 3>

<item 4>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 4>

<item 5>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 5>

<item 6>
The Private Eyes (1976 film)
The Private Eyes is a 1976 Hong Kong comedy film written, directed by and starring Michael Hui and co-starring his brothers, Samuel Hui and Ricky Hui, and also
starring Shih Kien and Richard Ng in his second film role.

 John Woo was the production designer and also co-director though he was uncredited.

 Sammo Hung served as the film's action director and Jackie Chan was also a stuntman.

 This is the third film of the Hui Brothers and it is the first film that established the Hui Brothers' comedies internationally.
</item 6>

<item 7>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 7>

<item 8>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 8>

<item 9>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 9>

<item 10>
Jon Paul Puno
Jon Paul Puno (born 1985) is an American filmmaker and classical crossover singer.

 As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).

 And also played a small role in the 2006 movie I Will Always Love You (film).
</item 10>
=== END ITEM id=99176d6932fabd62 ===

=== ITEM id=6e514b6ee359c5f5 ===
Gold sentences and facts:
Sentence 0: Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international management consultant.
  Fact 0.0: Eenasul Fateh was born on 3 April 1959.
  Fact 0.1: Eenasul Fateh's Bengali name is ঈনাসুল ফাতেহ.
  Fact 0.2: Eenasul Fateh is also known by his stage name Aladin.
  Fact 0.3: Eenasul Fateh is a Bangladeshi-British cultural practitioner.
  Fact 0.4: Eenasul Fateh is a magician.
  Fact 0.5: Eenasul Fateh is a live artist.
  Fact 0.6: Eenasul Fateh is a former international management consultant.
Sentence 1: Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans for improvement.
  Fact 1.0: Management consulting is the practice of helping organizations improve their performance.
  Fact 1.1: Management consulting operates primarily through the analysis of existing organizational problems.
  Fact 1.2: Management consulting operates primarily through the development of plans for improvement.

Retrieved items:
<item 1>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 1>

<item 2>
Graph query row:
person: Eenasul Fateh
occupation: ['cultural practitioner', 'magician', 'live artist', 'former international management consultant']
nationality: Bangladeshi-British
birth_date: 3 April 1959
death_date: None
relationship: None
connected_entity: None
</item 2>

<item 3>
Management consulting
Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational
problems and the development of plans for improvement.

 Organizations may draw upon the services of management consultants for a number of reasons, including gaining external (and presumably objective) advice and
access to the consultants' specialized expertise.
</item 3>

<item 4>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 4>

<item 5>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 5>

<item 6>
Eenasul Fateh
Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live
artist and former international management consultant.
</item 6>

<item 7>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 7>

<item 8>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 8>

<item 9>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 9>

<item 10>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 10>
=== END ITEM id=6e514b6ee359c5f5 ===

=== ITEM id=d21be409e6d6443f ===
Gold sentences and facts:
Sentence 0:  Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer Jimmy Chamberlin.
  Fact 0.0: The band was formed by Corgan and guitarist James Iha.
  Fact 0.1: The band was formed in Chicago, Illinois, in 1988.
  Fact 0.2: The band quickly gained steam with the addition of bassist D'arcy Wretzky.
  Fact 0.3: The band quickly gained steam with the addition of drummer Jimmy Chamberlin.
Sentence 1: Greek Fire is an American rock band from St. Louis, Missouri.
  Fact 1.0: Greek Fire is an American rock band.
  Fact 1.1: Greek Fire is from St. Louis, Missouri.

Retrieved items:
<item 1>
Greek Fire (band)
Greek Fire is an American rock band from St. Louis, Missouri.

 The band was formed in 2008 by members of Story of the Year and Maybe Today.

 Since formation, Greek Fire has released a self-titled EP, a single titled "Doesn't Matter Anyway", on August 16, 2011, they released their debut, full-length
album, "Deus Ex Machina", and have recently announced a new addition to the "Lost/Found" EPs titled "Broken" set to be released before "Found".
</item 1>

<item 2>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 2>

<item 3>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 3>

<item 4>
Zeitgeist (The Smashing Pumpkins album)
Zeitgeist is the seventh album by American alternative rock band The Smashing Pumpkins, released on July 10, 2007 in the United States and Canada.

 It was the first album The Smashing Pumpkins released after their 2000 disbandment and 2005 reunion.

 The album was produced by Roy Thomas Baker, Billy Corgan, Jimmy Chamberlin, and Terry Date.

 The album would be Chamberlin's last with the band before his departure in 2009.

 The album debuted strongly, but sales soon decreased, and critical reception was mixed.

 It was certified Gold in the United States on February 1, 2008.
</item 4>

<item 5>
Mike Byrne (musician)
Michael William Byrne (born February 6, 1990) is a drummer who was a member of the band The Smashing Pumpkins.

 When Pumpkins frontman Billy Corgan called for rehearsals to replace Jimmy Chamberlin, Byrne auditioned and was chosen out of thousands of applicants.

 He played drums in Sky Saxon tribute band Spirits in the Sky for 6 shows in August 2009, along with Corgan, Kerry Brown, Kevin Dippold, Mark Tulin, Linda
Strawberry, Ysanne Spevack, Mark Weitz, and Dave Navarro, and also performs drums on the ongoing Pumpkins 44-song project, "Teargarden by Kaleidyscope".

 As a part of "Teargarden", Mike contributed drumming duties on "Oceania", the Pumpkins' eighth full-length album.

 On April 17, 2010, Byrne played his first show under the Smashing Pumpkins moniker in celebration of Record Store Day.
</item 5>

<item 6>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 6>

<item 7>
List of the Smashing Pumpkins band members
The Smashing Pumpkins are an alternative rock band formed in Chicago, Illinois in 1988.

 The band was formed by guitarist/vocalist Billy Corgan and guitarist James Iha after the demise of Corgan's first band, The Marked.

 Since its inception, The Smashing Pumpkins has gone through multiple line-up changes, with Corgan the only consistent member.
</item 7>

<item 8>
1979 (song)
"1979" is a song by American alternative rock band the Smashing Pumpkins.

 Released in 1996 as the second single from their third studio album, "Mellon Collie and the Infinite Sadness", "1979" was written by frontman Billy Corgan, and
features loops and samples that were uncharacteristic of previous Smashing Pumpkins songs.

 The song was written as a coming of age story by Corgan.

 In the year 1979, Corgan was 12 and this is what he considered his transition into adolescence.

 The song was popular with critics and fans; Allmusic's Amy Hanson called it a "somewhat surprising hit".

 The song was nominated for the Record of the Year and Best Rock Performance by a Duo or Group with Vocal at the 39th Annual Grammy Awards, and won an MTV Video
Music Award for Best Alternative Video.

 In 2012, it was voted the second-best Smashing Pumpkins song by Rolling Stone magazine readers.
</item 8>

<item 9>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 9>

<item 10>
The Cult’s *High Octane Cult* compiled its singles but was issued without the band, whose members criticized its artwork and presentation. Smashing Pumpkins’
*The Aeroplane Flies High* (1996) was a successful five-disc, platinum box set; *Zeitgeist* (2007), their first post-reunion album, earned Gold despite mixed
reviews. Billy Corgan is the band’s enduring leader; drummer Jimmy Chamberlin departed in 2009 and returned in
</item 10>
=== END ITEM id=d21be409e6d6443f ===

=== ITEM id=8f33b1c5fc5e42ec ===
Gold sentences and facts:
Sentence 0: "Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".
  Fact 0.0: 'Collateral Damage' is the eleventh episode of the third season of the crime-thriller television series 'Millennium'.
  Fact 0.1: 'Millennium' is a crime-thriller television series.
  Fact 0.2: 'Collateral Damage' is in the third season of 'Millennium'.
Sentence 1:  "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.
  Fact 1.0: 'Collateral Damage' featured a guest appearance by O'Quinn.
  Fact 1.1: 'Collateral Damage' featured a guest appearance by Jacinda Barrett.
  Fact 1.2: 'Collateral Damage' featured a guest appearance by James Marsters.
  Fact 1.3: 'Collateral Damage' featured a guest appearance by radio host Art Bell.
  Fact 1.4: Art Bell appeared as himself in 'Collateral Damage'.
Sentence 2:  Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
  Fact 2.0: Art Bell's radio show 'Coast to Coast AM' was among Perry's inspirations for the script.
Sentence 3: Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.
  Fact 3.0: Coast to Coast AM is an American late-night radio talk show.
  Fact 3.1: Coast to Coast AM deals with a variety of topics.
Sentence 4:  The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
  Fact 4.0: Coast to Coast AM now airs seven nights a week.
  Fact 4.1: Coast to Coast AM airs from 1:00 a.m. to 5:00 a.m. Eastern Time Zone.

Retrieved items:
<item 1>
Psycho Circus World Tour
This ultimately happened only at the first concert at Dodger Stadium.

 Peter Criss later said that it didn't work out because the circus performers wanted equal billing and that some had even wanted to use KISS' backstage dressing
room.

 "2,000 Man" was played to bring in the new Millennium at the 1999/2000 New Year's Eve show at Vancouver, advertised at the time as being recorded for Alive IV.

 The Vancouver show was also notable as being the first time the original members had played non-original band era material live in concert – "I Love It Loud",
"Lick It Up" and "Heaven's on Fire" were added to the setlist and subsequently played on the Farewell Tour.

 "Forever" was listed on concert setlists at the Vancouver show but was not played.

 It was thought at the time it may have been intended as a Paul Stanley solo version prior to "Black Diamond".

 One notable show on the tour was the March 12 Bremen, Germany, show.

 After the opening song, Paul Stanley announced that the local fire marshall had banned Kiss from using any pyrotechnics during the show.
</item 1>

<item 2>
Night After Night with Allan Havey
Night After Night with Allan Havey was a late night talk show which aired on The Comedy Channel and, later, Comedy Central from 1989 to 1992.

 The show featured the stream of consciousness of host Allan Havey, and other departures from the standard late-night talk show format.

 Havey's sidekick was Nick Bakay, who moved on to act as sidekick on Dennis Miller's short-lived late night talk show.
</item 2>

<item 3>
Godzilla: Final Wars
Godzilla: Final Wars (ゴジラ ファイナルウォーズ , Gojira: Fainaru Wōzu ) is a 2004 Japanese kaiju film featuring Godzilla, produced and distributed by Toho.

 It is the 29th film in the "Godzilla" franchise, the 28th Godzilla film produced by Toho, and the sixth and final film in the Millennium series.

 The film is directed by Ryuhei Kitamura, written by Wataru Mimura and Isao Kiriyama, and stars Masahiro Matsuoka, Rei Kikukawa, Don Frye, Maki Mizuno, Kazuki
Kitamura, Kane Kosugi, Masakatsu Funaki, Kumi Mizuno, Kenji Sahara, Masami Nagasawa, Chihiro Otsuka, Shigeru Izumiya, Masato Ibu, Jun Kunimura and Akira
Takarada.

 It is also the last Godzilla film to be produced by any studio until "Godzilla" (2014) and the last Godzilla film to be produced by Toho until "Shin Godzilla"
(2016).
</item 3>

<item 4>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 4>

<item 5>
Collateral Damage (Millennium)
"Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".

 It premiered on the Fox network on January 22, 1999.

 The episode was written by Michael R. Perry, and directed by Thomas J. Wright.

 "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.

 Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
</item 5>

<item 6>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 6>

<item 7>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 7>

<item 8>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 8>

<item 9>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 9>

<item 10>
David Letterman
David Michael Letterman (born April 12, 1947) is an American television host, comedian, writer, and producer.

 He hosted a late night television talk show for 33 years, beginning with the February 1, 1982 debut of "Late Night with David Letterman" on NBC, and ending
with the May 20, 2015 broadcast of "Late Show with David Letterman" on CBS.

 In total, Letterman hosted 6,028 episodes of "Late Night" and "Late Show", surpassing friend and mentor Johnny Carson as the longest-serving late night talk
show host in American television history.

 In 1996 Letterman was ranked 45th on "TV Guide"' s 50 Greatest TV Stars of All Time.
</item 10>
=== END ITEM id=8f33b1c5fc5e42ec ===

=== ITEM id=208a2f61ce7c66ac ===
Gold sentences and facts:
Sentence 0: Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of the indie rock band Bloc Party.
  Fact 0.0: Kelechukwu 'Kele' Okereke was born on 13 October 1981.
  Fact 0.1: Kelechukwu 'Kele' Okereke is also known by the mononym Kele.
  Fact 0.2: Kelechukwu 'Kele' Okereke is a British musician.
  Fact 0.3: Kelechukwu 'Kele' Okereke is best known as the lead singer and rhythm guitarist of the indie rock band Bloc Party.
Sentence 1: Jon Hume (born 6 November 1983) is the lead singer of Evermore.
  Fact 1.0: Jon Hume was born on 6 November 1983.
  Fact 1.1: Jon Hume is the lead singer of Evermore.
Sentence 2:  Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
  Fact 2.0: Jon is the eldest of the three brothers.
  Fact 2.1: The three brothers make up the New Zealand band Evermore.
  Fact 2.2: Evermore is a New Zealand band.

Retrieved items:
<item 1>
Dann Hume
Dann Hume (born Daniel Benjamin Cobbe, 1 September 1987, Whangaparaoa, New Zealand) is a singer-songwriter and record producer.

 Hume is the youngest of three brothers (with Peter Hume and Jon Hume) who make up the alternative rock band, Evermore since 1999.
</item 1>

<item 2>
Evermore (Evermore album)
Evermore is the self-titled international debut album by New Zealand rock band Evermore, which was also released in Australia and New Zealand on 12 March 2010.

 The album is a collection of songs from the band's entire history, as well as three new songs including the single "Underground".
</item 2>

<item 3>
Between the Lines (Evermore song)
"Between the Lines" is the first single by Evermore, taken from their third studio album "".

 Evermore's Jon Hume said that "Between the Lines was the first song to come out of a search for a new musical experience as a band."

 It was released as a free download on Evermore's official website on 10 November 2008.
</item 3>

<item 4>
Light Surrounding You
"Light Surrounding You" is the second single by alternative rock band, Evermore, taken from their second studio album, "Real Life" (June 2006).

 It was written by Dann Hume, the group's drummer and lead singer and guitarists Jon Hume.

 The record is co-produced by Jon Hume.

 The song was released in October 2006 and peaked at No. 15 on the RIANZ Singles Chart in New Zealand and number one on the ARIA Singles Chart, making it
Evermore's most successful single in Australia.

 It was the first single by a New Zealand artist to top the Australian charts since "How Bizarre" by OMC in 1996.
</item 4>

<item 5>
Running (Evermore song)
"Running" is the lead single by alternative rock band Evermore, taken from their second studio album, "Real Life" (July 2006).

 It was written by backing vocalist and drummer, Dann Hume.

 It was issued on 3 June, ahead of the album.

 It was co-produced by Jon with John Alagía (Dave Matthews Band, John Mayer).

 It peaked at No. 4 on their native RIANZ Singles Chart in New Zealand and No. 5 on their adopted country of Australia's ARIA Singles Chart.
</item 5>

<item 6>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 6>

<item 7>
Kele Okereke
Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of
the indie rock band Bloc Party.
</item 7>

<item 8>
Jon Hume
Jon Hume (born 6 November 1983) is the lead singer of Evermore.

 He is a singer-songwriter, multi-instrumentalist, Record Producer and Studio owner.

 Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
</item 8>

<item 9>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 9>

<item 10>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 10>
=== END ITEM id=208a2f61ce7c66ac ===

=== ITEM id=92b4ce956c2ed780 ===
Gold sentences and facts:
Sentence 0:  The album was Prince's third to be released that year (along with "Chaos and Disorder" and the soundtrack album of the Spike Lee movie "Girl 6"), which made 1996 one of the most prolific years for material released by Prince.
  Fact 0.0: The album was Prince's third album released in 1996.
  Fact 0.1: One of Prince's other 1996 releases was "Chaos and Disorder".
  Fact 0.2: One of Prince's other 1996 releases was the soundtrack album of the Spike Lee movie "Girl 6".
  Fact 0.3: 1996 was one of the most prolific years for material released by Prince.
Sentence 1:  His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
  Fact 1.0: Prince's production company is 40 Acres and a Mule Filmworks.
  Fact 1.1: 40 Acres and a Mule Filmworks has produced over 35 films since 1983.

Retrieved items:
<item 1>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 1>

<item 2>
List of accolades received by The Sixth Sense
"The Sixth Sense" is an American supernatural thriller film, written and directed by M. Night Shyamalan.

 The film was released on August 6, 1999, grossing over $26,600,000 on its opening weekend and ranking first place at the box office.

 Overall the film grossed over $293,500,000 domestically and $672,800,000 worldwide, which is approximately 16.8 times its budget of $40 million.

 "The Sixth Sense" was well received by critics, with an approval rating of 85% from review aggregator Rotten Tomatoes.
</item 2>

<item 3>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 3>

<item 4>
Cannibal boom
Only four other cannibal films were made after 1981 until the fad's conclusion in 1988 with Antonio Climati's "Natura Contro".
</item 4>

<item 5>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 5>

<item 6>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 6>

<item 7>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 7>

<item 8>
The Doom Generation
The Doom Generation is a 1995 American dark comedy film written and directed by Gregg Araki.

 It stars James Duval, Rose McGowan, and Johnathon Schaech.

 The film follows two troubled teenage lovers Amy Blue (McGowan) and Jordan White (Duval) who pick up a young handsome drifter named Xavier Red (Schaech).

 After he accidentally kills a store's clerk, the trio embarks on a journey full of sex, violence, and people from Amy's past.

 Billed as "A Heterosexual Movie by Gregg Araki", "The Doom Generation" is the second film in the director's trilogy known as the "Teenage Apocalypse Trilogy",
the first being "Totally Fucked Up" (1993) and the last one "Nowhere" (1997).

 The characters of Amy Blue and Jordan White are based on the Mark Beyer comic strip "Amy and Jordan".
</item 8>

<item 9>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 9>

<item 10>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 10>
=== END ITEM id=92b4ce956c2ed780 ===

=== ITEM id=b9a5fd434057126a ===
Gold sentences and facts:
Sentence 0: Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National Remembrance.
  Fact 0.0: Kolejka is a board game.
  Fact 0.1: Kolejka is Polish for 'queue' or 'line (of people)'.
  Fact 0.2: Karol Madaj designed Kolejka.
  Fact 0.3: Kolejka was launched in February 2011.
  Fact 0.4: Kolejka was launched by Poland's Institute of National Remembrance.
Sentence 1: Ghettopoly is a parody of "Monopoly" released in 2003.
  Fact 1.0: Ghettopoly is a parody of Monopoly.
  Fact 1.1: Ghettopoly was released in 2003.

Retrieved items:
<item 1>
Plaid Hat Games
Plaid Hat Games is a United States-based board game studio.

 Plaid Hat Games was founded in 2009.

 Board game designer Colby Dauch formed a board game publishing company in order to release the companies first game, Summoner Wars.
</item 1>

<item 2>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 2>

<item 3>
Xeko
Xeko is a collectible card game revolving around endangered species.

 It was launched on Earthday 2006.

 It won the "Creative Child Magazine" 2006 Toy of the Year Award and the National Parenting Center's Seal of Approval in its first year.

 Four "Mission" sets have been released.

 "Mission: Costa Rica" and "Mission: Madagascar", based on biodiversity hotspots were released first.

 "Mission: Indonesia", was released in 2007, with the final release, "Mission: China", was released July 19, 2008.

 A total of thirty more missions were planned but never developed.
</item 3>

<item 4>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 4>

<item 5>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 5>

<item 6>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 6>

<item 7>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 7>

<item 8>
Ghettopoly
Ghettopoly is a parody of "Monopoly" released in 2003.

 Invented by David Chang, it uses "Monopoly"-like mechanics in the atmosphere of a caricaturized United States ghetto.
</item 8>

<item 9>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 9>

<item 10>
TriBond
TriBond is a board game that has sold over 3 million copies in 14 countries since its release in 1990.

 It requires players to determine a common bond between three subjects.

 It follows in the tradition of "Trivial Pursuit", "Outburst" and other adult boardgames that require a wide range of knowledge but "TriBond" requires some
problem solving ability as well.
</item 10>
=== END ITEM id=b9a5fd434057126a ===

=== ITEM id=841ac31b311858f8 ===
Gold sentences and facts:
Sentence 0:  He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
  Fact 0.0: The person in this sentence succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
Sentence 1: Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death in 1250.
  Fact 1.0: Eric IV is also known as Eric Ploughpenny.
  Fact 1.1: Eric IV is also known as Eric Plowpenny.
  Fact 1.2: Eric IV's Danish name is Erik Plovpenning.
  Fact 1.3: Eric IV's dates are 1216 to 10 August 1250.
  Fact 1.4: Eric IV was king of Denmark from 1241.
  Fact 1.5: Eric IV was king of Denmark until his death in 1250.

Retrieved items:
<item 1>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 1>

<item 2>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 2>

<item 3>
King Kang of Chu
King Kang of Chu (, died 545 BC) was from 559 to 545 BC the king of Chu, a major power during the Spring and Autumn period of ancient China.

 Born Xiong Zhao (), he succeeded his father King Gong of Chu.

 His reign was marked by constant wars with Jin, Chu's traditional enemy, and Wu, its new enemy.

 He ruled for 15 years and was succeeded by his son, Jia'ao, who would four years later be murdered by King Kang's younger brother King Ling of Chu.
</item 3>

<item 4>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 4>

<item 5>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 5>

<item 6>
Christopher I of Denmark
Christopher I (Danish: "Christoffer I" ) (1219 – 29 May 1259) was King of Denmark between 1252 and 1259.

 He was the son of Valdemar II of Denmark by his wife, Infanta Berengária of Portugal.

 He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.

 Christopher was elected King upon the death of his older brother Abel in the summer of 1252.

 He was crowned at Lund Cathedral on Christmas Day 1252.
</item 6>

<item 7>
Eric IV of Denmark
Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death
in 1250.

 He was the son of King Valdemar II by his wife, Berengaria of Portugal, and brother of King Abel and King Christopher I.
</item 7>

<item 8>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 8>

<item 9>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 9>

<item 10>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 10>
=== END ITEM id=841ac31b311858f8 ===

=== ITEM id=422310db68ec16df ===
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
Aulacodes templalis
Aulacodes templalis is a moth in the Crambidae family.

 It was described by Schaus in 1906.

 It is found in Brazil.
</item 1>

<item 2>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 2>

<item 3>
Agrotis orthogonia
The Pale Western Cutworm ("Agrotis orthogonia") is a moth of the Noctuidae family.

 It is found in North America, more specifically dry, semi-desert areas of western North America from southern Canada to California, ranging eastward nearly to
the eastern edge of the Great Plains.
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
Al Khor (city)
Al Khor is a coastal city in northern Qatar, located 50 kilometres north of the capital, Doha.

 It is the capital city of the municipality of Al Khor.

 The name of the city means creek in Arabic as the town is located on a creek.

 Al Khor is home to many employees of the oil industry due to its proximity to Qatar's northern oil and natural gas fields, and due to its proximity to the Ras
Laffan Industrial City.
</item 5>

<item 6>
Amka
Amka (Hebrew: עַמְקָה‎ ), also known in Arabic as Amqa (Arabic: عمقا‎ ‎ ), is a moshav in the Matte Asher Regional Council of Israel's Northern District, near
Acre.

 The location of the moshav roughly corresponds the former Arab village, depopulated during the 1948 Arab–Israeli War.

 Yemenite Jews, who arrived from the southern Arab country of Yemen, founded the village's successor Amka in 1949.

 In 2016 its population was 706 .
</item 6>

<item 7>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 7>

<item 8>
Ornithodoros savignyi
Because of its habit of feeding and dropping from its host, adult dispersal is limited, whereas larvae may remain attached to their hosts for several days.

 During its life cycle it will feed on multiple hosts between moults.
</item 8>

<item 9>
Aulacodes traversalis
Aulacodes traversalis is a moth in the Crambidae family.

 It was described by Harrison Gray Dyar Jr. in 1914.

 It is found in Panama.
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
=== END ITEM id=422310db68ec16df ===

=== ITEM id=295065c4dc901e46 ===
Gold sentences and facts:
Sentence 0: Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
  Fact 0.0: Giuseppe Fortunino Francesco Verdi was born on 9 or 10 October 1813.
  Fact 0.1: Giuseppe Fortunino Francesco Verdi died on 27 January 1901.
  Fact 0.2: Giuseppe Fortunino Francesco Verdi was an Italian opera composer.
Sentence 1: Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
  Fact 1.0: Charles Louis Ambroise Thomas was born on 5 August 1811.
  Fact 1.1: Charles Louis Ambroise Thomas died on 12 February 1896.
  Fact 1.2: Charles Louis Ambroise Thomas was a French composer.
  Fact 1.3: Charles Louis Ambroise Thomas is best known for his opera 'Mignon' (1866).
  Fact 1.4: Charles Louis Ambroise Thomas is best known for his opera 'Hamlet' (1868).
  Fact 1.5: The opera 'Hamlet' (1868) is after Shakespeare.
  Fact 1.6: Charles Louis Ambroise Thomas is best known as Director of the Conservatoire de Paris.
  Fact 1.7: Charles Louis Ambroise Thomas was Director of the Conservatoire de Paris from 1871 till his death.

Retrieved items:
<item 1>
List of major opera composers
This list provides a guide to opera composers, as determined by their presence on a majority of compiled lists of significant opera composers.

 (See the "Lists Consulted" section for full details.)

 The composers run from Jacopo Peri, who wrote the first ever opera in late 16th century Italy, to John Adams, one of the leading figures in the contemporary
operatic world.

 The brief accompanying notes offer an explanation as to why each composer has been considered major.

 Also included is a section about major women opera composers, compiled from the same lists.

 For an introduction to operatic history, see opera.

 The organisation of the list is by birthdate.
</item 1>

<item 2>
Teatro Giuseppe Verdi
Teatro Giuseppe Verdi (the Giuseppe Verdi Theatre) is a small opera house located in a wing of the Rocca dei Marchesi Pallavicino on the Piazza Giuseppe Verdi
in Busseto, Italy, a town closely associated with the life of the opera composer, Giuseppe Verdi.

 From the 13th century, the “rocca” or “fortress” was the family’s palace; it is now the city hall after being acquired by the municipality in 1856.

 The theatre opened on 15 August 1868 and seats 300.
</item 2>

<item 3>
Giuseppe Verdi (film)
Giuseppe Verdi is a 1938 Italian biographical film directed by Carmine Gallone and starring Fosco Giachetti, Gaby Morlay and Germana Paolieri.

 The film portrays the life of the composer Giuseppe Verdi (1813-1901).

 The casting of Giachetti as Verdi was intended to emphasise the composer's patriotism, as he had recently played patriotic roles in films such as "The White
Squadron".

 The film was made at the Cinecittà Studios in Rome.

 The film is also known by the alternative title The Life of Giuseppe Verdi.
</item 3>

<item 4>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 4>

<item 5>
Ferdinando Provesi
Ferdinando Angelo Maria Provesi (1770 – 1833) was a native of Parma, Italy.

 He was regarded as one of the greatest Italian opera composers of the era .

 Provesi is best known as being an early tutor of Giuseppe Verdi when he was the "Maestro di cappella" (master of music) at the St. Bartolomeo cathedral in
Busseto (the town very close to Le Roncole, the village where Verdi was born.)

 Provesi was also director of the municipal music school and local Philharmonic Society.

 He began teaching Verdi in 1824 when the future composer was 11 years old.
</item 5>

<item 6>
Ambroise Thomas
Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after
Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
</item 6>

<item 7>
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 7>

<item 8>
Verdi, the King of Melody
Giuseppe Verdi, released theatrically in the USA as The Life and Music of Giuseppe Verdi and on video as Verdi, the King of Melody, is a 1953 Italian
biographical film starring Pierre Cressoy and directed by Raffaello Matarazzo.

 It is based on adult life events of the composer Giuseppe Verdi.

 The film was a commercial success, grossing over 957 million lire at the Italian box office.
</item 8>

<item 9>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 9>

<item 10>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 10>
=== END ITEM id=295065c4dc901e46 ===

=== ITEM id=fbaa6b9427f3f2c4 ===
Gold sentences and facts:
Sentence 0: William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.
  Fact 0.0: William Dwight Schultz was born on November 24, 1947.
  Fact 0.1: William Dwight Schultz is an American actor.
  Fact 0.2: William Dwight Schultz is an American voice artist.
Sentence 1:  He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the Squirrel in "CatDog".
  Fact 1.0: William Dwight Schultz is well known in animation as the mad scientist Dr. Animo in the "Ben 10" series.
  Fact 1.1: William Dwight Schultz is well known in animation as Chef Mung Daal in the children's cartoon "Chowder".
  Fact 1.2: William Dwight Schultz is well known in animation as Eddie the Squirrel in "CatDog".
Sentence 2: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 2.0: Chowder is an American animated television series.
  Fact 2.1: Chowder was created by C. H. Greenblatt.
  Fact 2.2: Chowder was created for Cartoon Network.

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
Rodger Bumpass
Rodger Bumpass (born November 20, 1951) is an American actor and voice actor.

 He is best known for his long-running role as Squidward Tentacles on the American animated television series "SpongeBob SquarePants".

 He voices many other characters on the show, including Dr. Forrest and various anchovies.

 He also voiced The Chief in the animated series "Where on Earth Is Carmen Sandiego?

", and Mr. Besser, the school principal in the animated series "The Kids from Room 402".

 Bumpass has many other credits in animated films, animated television series, and video games.
</item 2>

<item 3>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 3>

<item 4>
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
</item 4>

<item 5>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 5>

<item 6>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 6>

<item 7>
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
</item 7>

<item 8>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 8>

<item 9>
Dan Castellaneta
Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.

 Noted for his long-running role as Homer Simpson on the animated television series "The Simpsons", he also voices many other characters for the show, including
Abraham "Grampa" Simpson, Barney Gumble, Krusty the Clown, Sideshow Mel, Groundskeeper Willie, Mayor Quimby and Hans Moleman.
</item 9>

<item 10>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 10>
=== END ITEM id=fbaa6b9427f3f2c4 ===
