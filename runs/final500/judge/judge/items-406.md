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

=== ITEM id=ca79ec8b27672c7a ===
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
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 1>

<item 2>
Martin Degville
Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit
single in 1986 with "Love Missile F1-11" – and six other EMI single releases.

 Sputnik was formed with ex-Generation X bassist Tony James.
</item 2>

<item 3>
Poco: The Songs of Richie Furay
Poco: The Songs of Richie Furay is a compilation album consisting of songs by Richie Furay during his tenure with the band Poco, released in 1980.
</item 3>

<item 4>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 4>

<item 5>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
</item 5>

<item 6>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 6>

<item 7>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 7>

<item 8>
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
</item 8>

<item 9>
Last Time Around
Last Time Around is the third and final album by folk rock band Buffalo Springfield.

 Released 1968, it featured Neil Young, Stephen Stills, Richie Furay, Dewey Martin, Bruce Palmer, and Jim Messina, who also worked as producer and a recording
engineer.
</item 9>

<item 10>
Poco
Poco is an American country rock band originally formed by Richie Furay, Jim Messina and Rusty Young.

 Formed following the demise of Buffalo Springfield in 1968, Poco was part of the first wave of the West Coast country rock genre.

 The title of their first album, "Pickin' Up the Pieces", is a reference to the break-up of Buffalo Springfield.

 Throughout the years Poco has performed in various groupings, and is still active.
</item 10>
=== END ITEM id=ca79ec8b27672c7a ===

=== ITEM id=6ab16c6c43e3ead0 ===
Gold sentences and facts:
Sentence 0: Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer, musician, director, actor, and comedian who holds dual British and American citizenship.
  Fact 0.0: Christopher Haden-Guest is the 5th Baron Haden-Guest.
  Fact 0.1: Christopher Haden-Guest was born on February 5, 1948.
  Fact 0.2: Christopher Haden-Guest is simply known as Christopher Guest.
  Fact 0.3: Christopher Guest is an English-American screenwriter.
  Fact 0.4: Christopher Guest is an English-American composer.
  Fact 0.5: Christopher Guest is an English-American musician.
  Fact 0.6: Christopher Guest is an English-American director.
  Fact 0.7: Christopher Guest is an English-American actor.
  Fact 0.8: Christopher Guest is an English-American comedian.
  Fact 0.9: Christopher Guest holds dual British and American citizenship.
Sentence 1:  Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.0: Christopher Guest is most widely known in Hollywood for having written his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.1: Christopher Guest is most widely known in Hollywood for having directed his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.2: Christopher Guest is most widely known in Hollywood for having starred in his series of comedy films shot in mock-documentary (mockumentary) style.
Sentence 2:  He also directed dark comedies, children's movies, and historical pictures.
  Fact 2.0: Christopher Guest directed dark comedies.
  Fact 2.1: Christopher Guest directed children's movies.
  Fact 2.2: Christopher Guest directed historical pictures.

Retrieved items:
<item 1>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 1>

<item 2>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 2>

<item 3>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 3>

<item 4>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 4>

<item 5>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 5>

<item 6>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 6>

<item 7>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 7>

<item 8>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 8>

<item 9>
Vegas Vacation
Vegas Vacation is a 1997 American comedy film directed by Stephen Kessler.

 It is the fourth installment in "National Lampoon"’s "Vacation" film series, and was written by Elisa Bell, based on a story by Bell and Bob Ducsay.

 The film stars Chevy Chase, Beverly D'Angelo and Randy Quaid, with Ethan Embry and Marisol Nichols as Griswold children Rusty and Audrey.

 The film opened at #4 at the box office and grossed over $36.4 million domestically.

 "Vegas Vacation" became the first theatrical "Vacation" film not to carry the National Lampoon label or a screenwriting credit from John Hughes.
</item 9>

<item 10>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 10>
=== END ITEM id=6ab16c6c43e3ead0 ===

=== ITEM id=2a76f10c590748e7 ===
Gold sentences and facts:
Sentence 0: Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.
  Fact 0.0: Cherie Ann Currie was born November 30, 1959.
  Fact 0.1: Cherie Ann Currie is an American musician.
  Fact 0.2: Cherie Ann Currie is an instrumentalist, singer, and songwriter.
  Fact 0.3: Cherie Ann Currie is an actress.
  Fact 0.4: Cherie Ann Currie is an artist.
Sentence 1: Steve Augeri (born January 30, 1959) is an American rock singer best known for his work with Tall Stories, Tyketto, and Journey.
  Fact 1.0: Steve Augeri was born January 30, 1959.
  Fact 1.1: Steve Augeri is an American rock singer.
  Fact 1.2: Steve Augeri is best known for his work with Tall Stories.
  Fact 1.3: Steve Augeri is best known for his work with Tyketto.
  Fact 1.4: Steve Augeri is best known for his work with Journey.

Retrieved items:
<item 1>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 1>

<item 2>
Steve Augeri
Steve Augeri (born January 30, 1959) is an American rock singer best known for his work with Tall Stories, Tyketto, and Journey.
</item 2>

<item 3>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 3>

<item 4>
Reverie (Cherie Currie album)
Reverie is the third full-length studio album by Cherie Currie.

 Released on iTunes March 16, 2015.

 Cherie released the CD version of this album June 5, 2015 on her eBay page cheriecurriedirect.

 There is a 35-year gap between Cherie's last full-length studio album, 1980's Messin' with the Boys (with Marie Currie), and 2015's Reverie.

 This is last studio album Kim Fowley produced before his death.

 Kim helped Cherie release this album to make amends with her after all the money he swindled her out of when she was in the Runaways and for releasing her and
Marie's music on Young and Wild without their approval.

 After Kim's death Cherie's son, Jake Hays, took over producing.
</item 4>

<item 5>
Marie Currie
Marie Michelle Currie (born November 30, 1959) is an American singer, songwriter, actress, and artist.

 Currie is best known for playing in a band with her identical twin Cherie Currie called Cherie & Marie Currie.

 Their song "Since You Been Gone" charted at number 95 on the US charts.

 Marie played Singing Maid Marie in "The Rosebud Beach Hotel" and is now a popular multi-media sculptress and artist.
</item 5>

<item 6>
Messin' with the Boys
Messin' with the Boys is the second post-Runaways album by Cherie Currie, released in 1980 For this album Currie worked with her identical twin sister, Marie
Currie.

 Their band was called "Cherie and Marie Currie".

 Marie was a guest vocalist on Cherie's first album, so Marie went on tour with Cherie to support her first album, "Beauty's Only Skin Deep".

 When Marie would join Cherie on stage to sing the encores the audience would go wild.

 So Cherie ran with the idea of two blonds are better than one.

 The idea paid off because "Messin' with the Boys" received more radio play than "Beauty's Only Skin Deep", and their song "Since You Been Gone" made it to 95
on US charts.

 The single "This Time" and the album Messin' with the Boys made the top 200 on U.S. charts.

 This makes Messin' with the Boys Cherie Currie's most successful album.

 "I Just Love the Feeling" originally surfaced on the 1974 album, "S.S.
</item 6>

<item 7>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 7>

<item 8>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 8>

<item 9>
Age of Chance
Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.

 They were perhaps most known for their mutant metallic cover of Prince's "Kiss" which topped the UK Indie Chart in 1986, and peaked at No. 50 in the UK Singles
Chart in January the following year.

 Despite signing for major label Virgin, and being favourites with the UK music press, they never enjoyed a major hit in the UK, although "Don't Get Mad… Get
Even" reached No. 8 on the US "Billboard" Hot Dance/Club Play chart.
</item 9>

<item 10>
Young and Wild (album)
Young and Wild is a compilation by Cherie & Marie Currie.

 This album has all 10 original tracks from Messin' with the Boys, six songs from Beauty's Only Skin Deep, three songs Cherie Currie sang with The Runaways, and
one new track co-written by Marie Currie, "Longer Than Forever".

 "Longer Than Forever" was the B side of the single "Since You Been Gone".
</item 10>
=== END ITEM id=2a76f10c590748e7 ===

=== ITEM id=a742aaa40048a6bf ===
Gold sentences and facts:
Sentence 0: The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between 1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
  Fact 0.0: The New York – New Jersey Line War is also known as the N.J. Line War.
  Fact 0.1: The New York – New Jersey Line War refers to a series of skirmishes and raids.
  Fact 0.2: The skirmishes and raids took place for over half a century.
  Fact 0.3: The skirmishes and raids took place between 1701 and 1765.
  Fact 0.4: The skirmishes and raids took place at the disputed border between two American colonies.
  Fact 0.5: The two American colonies were the Province of New York and the Province of New Jersey.
Sentence 1: The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.
  Fact 1.0: The Province of New York existed from 1664 to 1776.
  Fact 1.1: The Province of New York was a British proprietary colony.
  Fact 1.2: The Province of New York later became a royal colony.
  Fact 1.3: The Province of New York was on the northeast coast of North America.

Retrieved items:
<item 1>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 1>

<item 2>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 2>

<item 3>
Province of Pennsylvania
The Province of Pennsylvania, also known as the Pennsylvania Colony, was founded in English North America by William Penn on March 4, 1681 as dictated in a
royal charter granted by King Charles II.

 The name Pennsylvania, which translates roughly as "Penn's Woods", was created by combining the Penn surname (in honor of William's father, Admiral Sir William
Penn) with the Latin word "sylvania", meaning "forest land."

 The Province of Pennsylvania was one of the two major Restoration colonies, the other being the Province of Carolina.

 The proprietary colony's charter remained in the hands of the Penn family until the American Revolution, when the Commonwealth of Pennsylvania was created and
became one of the original thirteen states.
</item 3>

<item 4>
Province of New York
The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.

 As one of the Thirteen Colonies, New York achieved independence and worked with the others to found the United States.
</item 4>

<item 5>
New York – New Jersey Line War
The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between
1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
</item 5>

<item 6>
Canada under British rule
Canada first came under British rule with the Treaty of Paris (1763) which ceded New France, of which Canada was a part, to the British Empire.

 Gradually, other territories, colonies, and provinces that were part of British North America would be added to Canada.

 The Royal Proclamation of 1763 enlarged the colony of Canada under the name of the Province of Quebec, which with the Constitutional Act 1791 became known as
The Canadas.

 With the Act of Union 1840 Upper and Lower Canada were joined to become the United Province of Canada.

 Later, with Confederation in 1867, the British maritime colonies of New Brunswick and Nova Scotia were joined with the British colony of Canada to form the
Dominion of Canada, which was subsequently divided into four provinces, Ontario, Quebec, New Brunswick, and Nova Scotia.

 A number of other British colonies, such as Newfoundland and British Columbia, and large territories such as Rupert's Land initially remained outside of the
newly formed federation.

 Over time, the remaining colonies and territories within British North America came under the control of Canada until the current geographic extent of the
country was reached when Newfoundland and Labrador joined Canada in 1949.
</item 6>

<item 7>
Stamp Act Congress
The Stamp Act Congress or First Congress of the American Colonies was a meeting held between October 7 and 25, 1765 in New York City, consisting of
representatives from some of the British colonies in North America; it was the first gathering of elected representatives from several of the American colonies
to devise a unified protest against new British taxation.

 Parliament had passed the Stamp Act, which required the use of specially stamped paper for legal documents, playing cards, calendars, newspapers and dice for
virtually all business in the colonies, and was going into effect on November 1.
</item 7>

<item 8>
Charter colony
Charter colony is one of three classes of colonial government established in the 17th century English colonies in North America, the other classes being
proprietary colony and royal colony.

 The colonies of Rhode Island, Connecticut, and Massachusetts Bay were charter colonies.

 In a charter colony, Britain granted a charter to the colonial government establishing the rules under which the colony was to be governed.

 The charters of Rhode Island and Connecticut granted the colonists significantly more political liberty than other colonies.

 Rhode Island and Connecticut continued to use their colonial charters as their State constitutions after the American Revolution.
</item 8>

<item 9>
History of the New Jersey State Constitution
Originally, the state of New Jersey was a single British colony, the Province of New Jersey.

 After the English Civil War, Charles II assigned New Jersey as a proprietary colony to be held jointly by Sir George Carteret and John Berkeley, 1st Baron
Berkeley of Stratton.

 Eventually, the collection of land fees, or quit-rents, from colonists proved inadequate for colonial profitability.

 Sir George Carteret sold his share of the colony to the Quakers in 1673.

 Following the sale, the land was divided into East and West Jersey.

 In 1681, West Jersey adopted a constitution.

 In 1683, East Jersey adopted one as well.

 In 1702, the colonies were united again under Anne, Queen of Great Britain, and adopted a constitution in 1776.
</item 9>

<item 10>
Proprietary colony
A proprietary colony was a type of British colony mostly in North America and the Caribbean in the 17th century.

 In the British Empire, all land belonged to the ruler, and it was his prerogative to divide.

 Therefore, all colonial properties were partitioned by royal charter into one of four types: proprietary, royal, joint stock, or covenant.

 King Charles II used the proprietary solution to reward allies and focus his own attention on Britain itself.

 He offered his friends colonial charters which facilitated private investment and colonial self-government.

 The charters made the proprietor the effective ruler, albeit one ultimately responsible to English law and the king.

 Charles II gave New Netherland to his younger brother The Duke of York, who named it New York.

 He gave an area to William Penn who named it Pennsylvania.
</item 10>
=== END ITEM id=a742aaa40048a6bf ===

=== ITEM id=000bdb9d3451800d ===
Gold sentences and facts:
Sentence 0: Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil War.
  Fact 0.0: Am Rong was born in 1929.
  Fact 0.1: Am Rong died in May 1975.
  Fact 0.2: Am Rong was Cambodian.
  Fact 0.3: Am Rong was a soldier.
  Fact 0.4: Am Rong was a filmmaker.
  Fact 0.5: Am Rong acted as a spokesman on military matters for the Khmer Republic.
  Fact 0.6: Am Rong acted as a spokesman during the Cambodian Civil War.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is American.
  Fact 1.2: Ava Marie DuVernay is a director.
  Fact 1.3: Ava Marie DuVernay is a screenwriter.
  Fact 1.4: Ava Marie DuVernay is a film marketer.
  Fact 1.5: Ava Marie DuVernay is a film distributor.

Retrieved items:
<item 1>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 1>

<item 2>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 2>

<item 3>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 3>

<item 4>
Graph query row:
person: Ava DuVernay
birth_date: August 24, 1972
</item 4>

<item 5>
Am Rong
Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil
War.

 Western journalists commented on the irony of his name as he gave briefings which "painted a rosy picture of the increasingly desperate situation on the
ground" during the war.
</item 5>

<item 6>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 6>

<item 7>
A Wrinkle in Time (2018 film)
A Wrinkle in Time is an upcoming American science-fiction fantasy adventure film directed by Ava DuVernay and written by Jennifer Lee.

 It is based on the 1963 novel of the same name by Madeleine L'Engle.

 The film stars Oprah Winfrey, Reese Witherspoon, Mindy Kaling, Storm Reid, Zach Galifianakis and Chris Pine.

 Principal photography on the film began on November 2, 2016, in Los Angeles, California.
</item 7>

<item 8>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 8>

<item 9>
Emayatzy Corinealdi
Emayatzy Evett Corinealdi ( ; born January 14, 1980) is an American actress.

 She is best known for her leading role in the Ava DuVernay film "Middle of Nowhere" (2012).
</item 9>

<item 10>
Graph query row:
person: Am Rong
birth_date: 1929
</item 10>
=== END ITEM id=000bdb9d3451800d ===

=== ITEM id=311fd5511af86826 ===
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
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 1>

<item 2>
Wild Fire (novel)
Wild Fire is a 2006 novel by American author, Nelson DeMille.

 It is the fourth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to Night Fall and takes place approximately one year later.

 Wild Fire is followed by DeMille’s 2010 novel, The Lion.
</item 2>

<item 3>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 3>

<item 4>
The Lion's Game
The Lion's Game is a 2000 novel by American author Nelson DeMille.

 It is the second of DeMille's novels to feature the detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New
York City.

 The 2000 novel The Lion’s Game is the sequel to Plum Island and is followed by the 2004 book Night Fall.

 The book also briefly mentions events from other DeMille novels like "The Charm School" and "The Gold Coast", that aren't strictly part of the Corey-universe.
</item 4>

<item 5>
George Orwell
Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.

 His work is marked by lucid prose, awareness of social injustice, opposition to totalitarianism, and outspoken support of democratic socialism.
</item 5>

<item 6>
The Panther (novel)
The Panther is a 2012 novel by American author Nelson DeMille.

 It is the sixth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "The Lion".

 The Panther is followed by DeMille’s 2015 novel, "Radiant Angel".

 Also featured in this novel is DeMille’s other fictional character, Paul Brenner, who appears in The General's Daughter and Up Country.
</item 6>

<item 7>
George Orwell: A Life in Pictures
George Orwell: A Life in Pictures is a 2003 BBC Television docudrama telling the life story of the British author George Orwell.

 Chris Langham plays the part of Orwell.

 No surviving sound recordings or video of the real George Orwell have been found.
</item 7>

<item 8>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 8>

<item 9>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 9>

<item 10>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 10>
=== END ITEM id=311fd5511af86826 ===

=== ITEM id=94d390be3f5aa39a ===
Gold sentences and facts:
Sentence 0: Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.
  Fact 0.0: Eriogonum is the scientific name for a genus of flowering plants.
  Fact 0.1: The genus Eriogonum is in the family Polygonaceae.
Sentence 1: Aloinopsis is a genus of ice plants from South Africa.
  Fact 1.0: Aloinopsis is a genus of ice plants.
  Fact 1.1: The Aloinopsis ice plants come from South Africa.

Retrieved items:
<item 1>
Alismataceae
The water-plantains (Alismataceae) are a family of flowering plants, comprising 11 genera and between 85 and 95 species.

 The family has a cosmopolitan distribution, with the greatest number of species in temperate regions of the Northern Hemisphere.

 Most of the species are herbaceous aquatic plants growing in marshes and ponds.
</item 1>

<item 2>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 2>

<item 3>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 3>

<item 4>
Eriogonum heracleoides
Eriogonum heracleoides (common names parsnipflower buckwheat, whorled buckwheat, and Wyeth buckwheat) has many flowering clusters that are cream or off-white in
color.

 Its usual habitat is rocky areas such as sagebrush deserts and Ponderosa pine forests.

 Parsnipflower buckwheat is in the genus "Eriogonum" and the family Polygonaceae which is a family of plants known as the "knotweed family".

 It inhabits much of the western part of the United States.
</item 4>

<item 5>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 5>

<item 6>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 6>

<item 7>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 7>

<item 8>
Aloinopsis
Aloinopsis is a genus of ice plants from South Africa.
</item 8>

<item 9>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 9>

<item 10>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 10>
=== END ITEM id=94d390be3f5aa39a ===

=== ITEM id=a8d32beaf6765a78 ===
Gold sentences and facts:
Sentence 0: "Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".
  Fact 0.0: 'Empty Orchestra' is the fourth episode of the third series of Inside No. 9.
  Fact 0.1: Inside No. 9 is a British dark comedy anthology television programme.
Sentence 1:  Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.
  Fact 1.0: 'Empty Orchestra' was written by Steve Pemberton.
  Fact 1.1: 'Empty Orchestra' was written by Reece Shearsmith.
  Fact 1.2: 'Empty Orchestra' was directed by Guillem Morales.
  Fact 1.3: 'Empty Orchestra' first aired on 7 March 2017.
  Fact 1.4: 'Empty Orchestra' first aired on BBC 2.
Sentence 2:  The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).
  Fact 2.0: The episode 'Empty Orchestra' is set in a karaoke booth.
  Fact 2.1: The episode 'Empty Orchestra' follows a group of work colleagues.
  Fact 2.2: The work colleague Greg is shown as Shearsmith.
  Fact 2.3: The work colleague Fran is shown as Sarah Hadland.
  Fact 2.4: The work colleague Connie is shown as Tamzin Outhwaite.
  Fact 2.5: The work colleague Janet is shown as Emily Howlett.
  Fact 2.6: The work colleague Duane is shown as Javone Prince.
  Fact 2.7: The colleagues are celebrating the promotion of Roger.
  Fact 2.8: Roger is shown as Pemberton.
Sentence 3:  Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and television productions, including army series "Red Cap" and crime drama "New Tricks".
  Fact 3.0: The performer described in this sentence came to national notice playing Melanie Owen in the BBC One soap opera EastEnders from 1998 until 2002.
  Fact 3.1: The performer has since starred in theatre productions.
  Fact 3.2: The performer has since starred in television productions.
  Fact 3.3: The performer has starred in the army series Red Cap.
  Fact 3.4: The performer has starred in the crime drama New Tricks.

Retrieved items:
<item 1>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 1>

<item 2>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 2>

<item 3>
Tamzin Outhwaite
Tamzin Maria Outhwaite ( ; born 5 November 1970) is a British actress from London.

 Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and
television productions, including army series "Red Cap" and crime drama "New Tricks".
</item 3>

<item 4>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 4>

<item 5>
Radio Cape Cod
Radio Cape Cod is a 2008 film directed and produced by Andrew Silver.

 It stars Tamzin Outhwaite as a radio interviewer coming to terms with the loss of her husband and a new love entering her life.

 She also must cope with her teenage daughter, played by Tamzin Merchant, who is experiencing her first love.

 The film also stars O. T. Fagbenle.

 The film won several festival awards.
</item 5>

<item 6>
Empty Orchestra
"Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".

 Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.

 The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily
Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).

 Rebekah Hinds also stars.
</item 6>

<item 7>
Out of Control (2002 film)
Out of Control is a 2002 British television film written and directed for BBC by Dominic Savage.

 It stars Tamzin Outhwaite as Shelley Richards, an impoverished single mother whose son, Dean (Danny Young), is involved in a string of crimes that lead to his
incarceration in a young offenders institute (YOI).

 David Morrissey plays prison warden Mike, who tries to keep Dean out of trouble but has difficultly watching him all the time.

 After continuous bullying at the YOI, Dean is no longer able to cope and makes a fatal decision.

 It is the third in a loose trilogy of films by Savage about social deprivation, following "Nice Girl" (2000) and "When I Was Twelve" (2001).
</item 7>

<item 8>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 8>

<item 9>
Melanie Owen
Melanie Jane "Mel" Owen (also Healy and Beale) is a fictional character from the BBC soap opera "EastEnders", played by Tamzin Outhwaite.

 She appeared from 19 October 1998 to 12 April 2002 when Outhwaite opted to leave the role.

 Mel was introduced by executive producer Matthew Robinson and became a prominent character during her tenure in the soap.
</item 9>

<item 10>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 10>
=== END ITEM id=a8d32beaf6765a78 ===

=== ITEM id=99b077b9500d1292 ===
Gold sentences and facts:
Sentence 0:  There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.
  Fact 0.0: There is no optimal strategy involved in the game.
  Fact 0.1: Play in the game entirely rests upon a random drawing of the cards.
Sentence 1: Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.
  Fact 1.0: Diplomacy is a strategic board game.
  Fact 1.1: Diplomacy was created by Allan B. Calhamer.
  Fact 1.2: Diplomacy was created in 1954.
  Fact 1.3: Diplomacy was released commercially in 1959.
Sentence 2:  Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.
  Fact 2.0: Diplomacy's main distinctions from most board wargames include its negotiation phases.
  Fact 2.1: In Diplomacy's negotiation phases, players spend much of their time forming and betraying alliances with other players.
  Fact 2.2: In Diplomacy's negotiation phases, players spend much of their time forming beneficial strategies.
  Fact 2.3: Diplomacy's main distinctions from most board wargames include the absence of dice and other game elements that produce random effects.

Retrieved items:
<item 1>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
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
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 3>

<item 4>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 4>

<item 5>
Diplomacy (game)
Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.

 Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players
and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.

 Set in Europe before the beginning of World War I, "Diplomacy" is played by two to seven players, each controlling the armed forces of a major European power
(or, with fewer players, multiple powers).

 Each player aims to move his or her few starting units and defeat those of others to win possession of a majority of strategic cities and provinces marked as
"supply centers" on the map; these supply centers allow players who control them to produce more units.
</item 5>

<item 6>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 6>

<item 7>
Simultaneous action selection
Simultaneous action selection, or SAS, is a game mechanic that occurs when players of a game take action (such as moving their pieces) at the same time.

 An example of a game that uses this type of movement is the game Diplomacy.

 Typically, a "secret yet binding" method of committing to one's move is necessary, so that as players' moves are revealed and implemented, others do not change
their moves in light of the new information.

 Thus, in Diplomacy, players write down their moves and then reveal them simultaneously.

 Because no player gets the first move, this potentially arbitrary source of advantage is not present.

 It is also possible for simultaneous movement games to proceed relatively quickly, because players are acting at the same time, rather than waiting for their
turn.

 Simultaneous action selection is easily implemented in card games such as Apples to Apples in which players simply select cards and throw them face-down into
the center.
</item 7>

<item 8>
TriBond (1990) is a knowledge-and-reasoning board game sold in 14 countries. Tapp-Tarock, an Austrian three-player tarot game recorded in 1821, was once popular
in Viennese cafés. Three-card brag, a 16th-century British bluffing game descended from Primero, helped inspire poker. Triomphe, a late-15th-century
trick-taking game, introduced randomly chosen trumps and inspired many later games. Ombre
</item 8>

<item 9>
Icehouse pyramids are stackable game pieces invented by Andrew Looney and John Cooper in 1987; Treehouse uses a die to guide moves matching players’
configurations to a shared one. Awithlaknannai Mosona is a Zuni Alquerque-related capture game documented in 1907. Alhambra (2003) is Dirk Henn’s Arabian-themed
tile game. Brunei’s Pasang scores captured tokens; Punjab’s Sher-bakar is a tiger hunt
</item 9>

<item 10>
The entries cover games across several eras and genres: Japan-only 1989 Famicom game *A Week of Garfield*, based on the comic; *Kill Doctor Lucky*, a humorous
1996 board game that won a 1997 Origins Award; Sid Sackson’s dice-risk racing game *Can’t Stop* (1980; reprinted 2007, iOS 2012); 2003 light war game *Attack!*,
also an Origins Award winner; the luck-based
</item 10>
=== END ITEM id=99b077b9500d1292 ===

=== ITEM id=71e514b31149b443 ===
Gold sentences and facts:
Sentence 0:  Պողոսյան , born July 15, 1990) is an American professional ice hockey defenseman currently playing for the Buffalo Sabres.
  Fact 0.0: Պողոսյան was born on July 15, 1990.
  Fact 0.1: Պողոսյան is an American professional ice hockey defenseman.
  Fact 0.2: Պողոսյան is currently playing for the Buffalo Sabres.
Sentence 1: Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record producer.
  Fact 1.0: The Armenian name of Daron Vartan Malakian is Տարօն/Տարոն Վարդան Մալաքեան.
  Fact 1.1: Daron Vartan Malakian was born on July 18, 1975.
  Fact 1.2: Daron Vartan Malakian is an Armenian-American singer-songwriter.
  Fact 1.3: Daron Vartan Malakian is a multi-instrumentalist.
  Fact 1.4: Daron Vartan Malakian is a record producer.

Retrieved items:
<item 1>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 1>

<item 2>
Zach Bogosian
Zachary M. Bogosian (Armenian: Զաքարի Մ.

 Պողոսյան , born July 15, 1990) is an American professional ice hockey defenseman currently playing for the Buffalo Sabres.

 Bogosian attended Cushing Academy in Massachusetts before he joined the Peterborough Petes of the Ontario Hockey League (OHL) in 2006.

 He played two seasons in the OHL, and was nominated for the Red Tilson Trophy as the league's most outstanding player in his second season.

 Bogosian was regarded as a complete, physical defenseman who could contribute on both offense and defense; he was rated as one of the top players heading into
the 2008 NHL Entry Draft, where the Atlanta Thrashers selected him third overall.

 He signed a contract with the team a few weeks after the draft and began the season with the Thrashers, though he missed several weeks of his rookie year due
to injury.

 In his second season, he tied a team record for goals by a defenseman.
</item 2>

<item 3>
Daryl Palumbo
Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.

 He is the frontman of the bands Glassjaw, Head Automatica and Color Film.

 As a youth he was a member of the Long Island straight edge band "XbustedX."

 In 1993 he met guitarist Justin Beck, who would become his good friend.

 Together they formed Glassjaw.
</item 3>

<item 4>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 4>

<item 5>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 5>

<item 6>
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
</item 6>

<item 7>
Dave Kneebone
Dave Kneebone is an American producer.

 Along with Eric Wareheim and Tim Heidecker he runs Abso Lutely Productions which produces a variety of television and films.

 His role has been described as the business chief and "straight man" at Abso Lutely.

 Kneebone has worked as a producer on a variety of television shows including "Comedy Bang!

 Bang!

", "Nathan for You", and "Tim and Eric Awesome Show, Great Job!

".

 He is also listed a producer on the feature film "Tim and Eric's Billion Dollar Movie".
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
Zach Bogosian
In his second season, he tied a team record for goals by a defenseman.

 Bogosian first played in an international tournament when he joined the American national team at the 2009 IIHF World Championship.
</item 9>

<item 10>
Bad Acid Trip
Bad Acid Trip is an American metal band, currently signed to System of a Down lead singer Serj Tankian's record label, Serjical Strike Records.

 Their 2004 album, "Lynch the Weirdo", was produced by Daron Malakian, another System of a Down member.

 In 2006 they performed on the second stage at the Ozzfest summer tour.

 The band has three full-length releases ("For the Weird by the Weird", "Lynch the Weirdo", and "Humanly Possible") and multiple EPs.
</item 10>
=== END ITEM id=71e514b31149b443 ===

=== ITEM id=85208f1548b4a15a ===
Gold sentences and facts:
Sentence 0: Samsung Galaxy S7 and Samsung Galaxy S7 Edge are Android smartphones manufactured and marketed by Samsung Electronics.
  Fact 0.0: Samsung Galaxy S7 is an Android smartphone.
  Fact 0.1: Samsung Galaxy S7 Edge is an Android smartphone.
  Fact 0.2: Samsung Galaxy S7 and S7 Edge are manufactured by Samsung Electronics.
  Fact 0.3: Samsung Galaxy S7 and S7 Edge are marketed by Samsung Electronics.
Sentence 1:  The S7 series serves as the successor to the Galaxy S6, S6 Edge and S6 Edge+ released in 2015.
  Fact 1.0: The S7 series is the successor to the Galaxy S6.
  Fact 1.1: The S7 series is the successor to the Galaxy S6 Edge.
  Fact 1.2: The S7 series is the successor to the Galaxy S6 Edge+.
  Fact 1.3: The Galaxy S6, S6 Edge and S6 Edge+ were released in 2015.
Sentence 2: Samsung Galaxy S6, Samsung Galaxy S6 Edge, and Samsung Galaxy S6 Edge+ are Android smartphones manufactured and marketed by Samsung Electronics.
  Fact 2.0: Samsung Galaxy S6 is an Android smartphone.
  Fact 2.1: Samsung Galaxy S6 Edge is an Android smartphone.
  Fact 2.2: Samsung Galaxy S6 Edge+ is an Android smartphone.
  Fact 2.3: Samsung Galaxy S6, S6 Edge and S6 Edge+ are manufactured by Samsung Electronics.
  Fact 2.4: Samsung Galaxy S6, S6 Edge and S6 Edge+ are marketed by Samsung Electronics.
Sentence 3:  The S6 line serves as a successor to the Galaxy S5.
  Fact 3.0: The S6 line is a successor to the Galaxy S5.
Sentence 4:  The S6 and S6 Edge smartphones were officially unveiled in the first "Samsung Unpacked 2015" event at Mobile World Congress on 1 March 2015, while the bigger S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5 in the second "Samsung Unpacked 2015" event at New York on 13 August 2015.
  Fact 4.0: The S6 was officially unveiled in the first Samsung Unpacked 2015 event.
  Fact 4.1: The S6 Edge was officially unveiled in the first Samsung Unpacked 2015 event.
  Fact 4.2: The first Samsung Unpacked 2015 event was held at Mobile World Congress on 1 March 2015.
  Fact 4.3: The S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5.
  Fact 4.4: The S6 Edge+ was officially unveiled in the second Samsung Unpacked 2015 event.
  Fact 4.5: The second Samsung Unpacked 2015 event was held in New York on 13 August 2015.

Retrieved items:
<item 1>
Samsung SGH-D500
The SGH-D500 has been succeeded by the Samsung SGH-D600.

 There is a very similar variant handset, the SGH-D500E.
</item 1>

<item 2>
Samsung Ativ S
The Samsung ATIV S is a touchscreen, slate smartphone manufactured by Samsung Electronics running the Windows Phone 8 operating system, upgradeable to Windows
Phone 8.1.

 The ATIV S is Samsung's first Windows Phone 8 device, and one of the first devices under its ATIV series of Microsoft Windows-based products.

 It was shown at the IFA 2012 in Berlin and was the first Windows Phone 8 device to be officially unveiled, ahead of the Nokia Lumia 920.
</item 2>

<item 3>
Samsung Galaxy S6
Samsung Galaxy S6, Samsung Galaxy S6 Edge, and Samsung Galaxy S6 Edge+ are Android smartphones manufactured and marketed by Samsung Electronics.

 The S6 line serves as a successor to the Galaxy S5.

 The S6 and S6 Edge smartphones were officially unveiled in the first "Samsung Unpacked 2015" event at Mobile World Congress on 1 March 2015, while the bigger
S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5 in the second "Samsung Unpacked 2015" event at New York on 13 August 2015.

 Alongside the S6, Samsung also unveiled the S6 Edge (and later on the bigger S6 Edge+), a variant whose screen is wrapped along the sides of the device; the
curvature is usable for several additional features.

 The Galaxy S6 and S6 Edge were released on 10 April 2015 in 20 countries while the S6 Edge+ was released on 21 August 2015 in 20 countries.
</item 3>

<item 4>
Samsung Galaxy Gear
The Samsung Galaxy Gear is a smartwatch produced by Samsung Electronics in the Samsung Gear family of devices.

 Unveiled during a "Samsung Unpacked" event in Berlin, Germany on September 4, 2013, the device serves as a companion for all Samsung Galaxy smartphones and
tablets which run Android 4.3 or newer; it was released on September 25, 2013.

 Originally released as an Android-based device, Samsung replaced the operating system with Tizen through the May 2014 software update.
</item 4>

<item 5>
Samsung Galaxy Tab 8.9
The Samsung Galaxy Tab 8.9 is an Android-based tablet computer designed and manufactured by Samsung, introduced on 22 March 2011 at CTIA wireless convention in
its Samsung Unpacked event in Orlando.

 It is part of the Samsung Galaxy Tab series, and features an 8.9-inch display and a 1 GHz dual-core Nvidia Tegra 2 processor.
</item 5>

<item 6>
Samsung SGH-T919
The Samsung Behold SGH-T919 is a touch-screen, 3G candybar-style mobile phone introduced by Samsung late in 2008.

 The Behold is one of the first Samsung mobile phones released to have a touch-screen along with the Samsung Omnia, the Samsung Instinct, the Samsung Eternity,
and the Samsung Tocco.

 The Behold is the "American" release of Samsung Tocco, which is only sold in Europe, Asia and Australia so the features and design of both of these phones are
similar.

 Some of the Behold's biggest competitors were Apple's iPhone, T-Mobile G1 (HTC Dream), LG Dare, Samsung Instinct, and Samsung Eternity.

 It was only available only to customers of T-Mobile.
</item 6>

<item 7>
Samsung Galaxy S Duos 2
Samsung Galaxy S Duos 2 is a version of Samsung Galaxy Trend Plus with dual SIM Android smartphone, produced and marketed by Samsung Electronics, which serves
as an immediate successor to the original Galaxy S Duos, which is made in second quarter of 2012.

 Unveiled on 30 November 2013 at some Asian markets, it was initially released on December 5, 2013 in several countries.

 In contrast with other dual sim Samsung models, this phone is a part of the high-end "S" series, this is why it is marketed as a part of the "Galaxy S" family.

 Despite being called as a direct successor to the original S Duos phone, external and physical design of the model is identical to the original model, placing
emphasis on internal upgrades such as upgraded processor and updated Operating System software.
</item 7>

<item 8>
Samsung Galaxy S7
Samsung Galaxy S7 and Samsung Galaxy S7 Edge are Android smartphones manufactured and marketed by Samsung Electronics.

 The S7 series serves as the successor to the Galaxy S6, S6 Edge and S6 Edge+ released in 2015.

 The phones were officially unveiled on 21 February 2016, during a Samsung press conference at Mobile World Congress, with a European and North American release
on 11 March 2016.
</item 8>

<item 9>
Samsung REX
The Samsung REX series is a range of Samsung smartphones and feature phones, ranging from low to mid-end phones, running a Java-based operating system with
Samsung's TouchWiz user interface.

 It is currently one of the Samsung line of devices that are aimed at the budget phone market alongside lower end Samsung Galaxy devices.

 Its main competition includes the Nokia Asha range and Firefox OS.
</item 9>

<item 10>
Samsung SGH-E900
The Samsung E900, introduced in 2006, is a high-end mobile phone and is derived from Samsung's D500-D600-D800 series of slide phones.

 It is of a slide-up design and has touch-sensitive keys, similar to the LG Chocolate (KG800).

 It was one of the most popular phones at the time but was discontinued late 2007 when buyers then turned their attention to the Samsung D900i, followed by the
Samsung U600, which was the slimmest phone by Samsung with a thickness of 1.09 cm until the E840 was released.
</item 10>
=== END ITEM id=85208f1548b4a15a ===
