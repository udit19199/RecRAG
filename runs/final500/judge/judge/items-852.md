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

=== ITEM id=64d616836f8a47e4 ===
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
Steve Miller recorded an unreleased demo of this song 's music with the lyrics of his other song , Take the Money and Run . The 2012 song Diese Tage with German
and English lyrics by KRIS feat Dante Thomas uses the chorus of `` The Joker '' . Country rapper Colt Ford sampled this song for his song `` Gangsta of Love ''
. `` Wow Wow '' by Neil Cicierega from the album `` Mouth Moods '' mixes `` The Joker '' with Will Smith 's `` Wild Wild West '' , `` All Star '' by Smash Mouth
and `` Cowboy '' by Kid Rock . In popular culture ( edit ) In 1996 , `` The Joker '' inspired a comedy film , The Pompatus of Love . References and footnotes (
edit ) ^ Jump up to : `` Steve Miller Band > Charts & Awards > Billboard Singles '' . allmusic . Retrieved October 5 , 2009 . ^ Jump up to : `` Chart Stats -
The Steve Miller Band - The Joker '' . Retrieved October 5 , 2009 . Jump up ^ `` irishcharts.ie search results '' .
</item 1>

<item 2>
The Joker (Steve Miller Band song)
The song topped the US Billboard Hot 100 in early 1974 . More than 16 years later , in September 1990 , it reached number one in the UK Singles Chart for two
weeks after being used in `` Great Deal '' , a Hugh Johnson - directed television advertisement for Levi 's , thus holding the record for the longest gap
between transatlantic chart - toppers . This reissue of `` The Joker '' also topped the Irish Singles Chart , the New Zealand RIANZ Singles Chart , the Dutch
Nationale Top 100 and the Dutch Top 40 . The first line of the lyrics is a reference to the song `` Space Cowboy '' from Miller 's Brave New World album .
Following lines refer to two other songs : `` Gangster of Love '' from Sailor and `` Enter Maurice '' from Recall the Beginning ... A Journey from Eden .
</item 2>

<item 3>
The Joker (Steve Miller Band song)
Retrieved October 6 , 2009 . Jump up ^ `` The Irish Charts - search result '' . Retrieved May 16 , 2016 . Jump up ^ `` norwegiancharts.com - Steve Miller Band -
The Joker '' . Retrieved October 6 , 2009 . Jump up ^ `` swedishcharts.com - Steve Miller Band - The Joker '' . Retrieved October 6 , 2009 . Jump up ^ `` Steve
Miller Band - The Joker - hitparade.ch '' ( in German ) . Retrieved October 6 , 2009 . Jump up ^ `` End of Year Charts 1991 '' . Recorded Music NZ . Retrieved
December 3 , 2017 . Jump up ^ `` Chart Stats - Fatboy Slim - The Joker '' . Retrieved October 13 , 2009 . Jump up ^ `` Smashing Pumpkins dot com '' . Retrieved
September 28 , 2009 . Jump up ^ `` The Howard Stern Radio Show '' . New York . 17 Apr 2013 . Sirius XM Satellite Radio . Howard 100 .
</item 3>

<item 4>
The Joker (Steve Miller Band song)
The Joker ( Steve Miller Band song ) - wikipedia The Joker ( Steve Miller Band song ) Jump to : navigation , search `` The Joker ( song ) '' redirects here .
For other songs with the same title , see Joker ( disambiguation ) . `` The Joker '' Cover of the 1973 single Single by Steve Miller Band from the album The
Joker B - side `` ' see track listings ' '' Released October 1973 Format 7 '' Single , 12 '' single , Cassette single ( Original Releases ) , CD single and
Digital download ( Post Releases ) . Genre Rock , blues rock , southern rock Length 4 : 26 ( Album version ) ; 3 : 35 ( 7 '' version / Radio edit ) and 6 : 20 (
Extended version ) Label Capitol Records ; EMI - Odeon ; London Records , Parlophone Records , EMI Bovema , Odeon Records .
</item 4>

<item 5>
The Joker (Steve Miller Band song)
Formats and track listings ( edit ) 7 '' single ( 1973 ) `` The Joker '' -- 3 : 36 `` Something to Believe In '' -- 4 : 40 7 '' single ( 1983 - live version )
`` The Joker '' ( live ) -- 2 : 55 `` Take the Money and Run '' ( live ) -- 3 : 49 7 '' single ( 1990 ) `` The Joker '' ( single version ) -- 3 : 34 `` Do n't
Let Nobody Turn You Around '' -- 2 : 27 12 '' maxi ( 1990 ) `` The Joker '' ( LP version ) -- 4 : 22 `` Do n't Let Nobody Turn You Around '' -- 2 : 27 `` Shu Ba
Da Du Ma Ma Ma '' -- 5 : 39 CD maxi ( 1990 ) `` The Joker '' ( single version ) -- 3 : 34 `` Do n't Let Nobody Turn You Around '' -- 2 : 27 `` Shu Ba Da Du Ma
Ma Ma Ma '' -- 3 : 33 `` Living in the U.S.A. ''
</item 5>

<item 6>
The Joker (Steve Miller Band song)
( live ) ( 1983 ) String Module Error : Match not found1983 `` Take the Money and Run '' ( live ) / `` The Joker '' ( live ) ( 1983 ) String Module Error :
Match not found1983 `` Shangri - La '' ( 1984 ) Shangri - La1984 `` The Joker '' ( 1990 , reissue ) Cover of the 1990 reissue Steve Miller Band singles
chronology `` Ya Ya '' ( 1988 ) Ya Ya1988 `` The Joker '' ( 1990 ) The Joker1990 `` Wide River '' ( 1993 ) Wide River1993 `` The Joker '' is a song by the Steve
Miller Band from their 1973 album The Joker . It is one of two Steve Miller Band songs that feature the nonce word `` pompatus '' . The song topped the US
Billboard Hot 100 in early 1974 .
</item 6>

<item 7>
The Joker (Steve Miller Band song)
Songwriter ( s ) Eddie Curtis , Ahmet Ertegün , Steve Miller Producer ( s ) Steve Miller Steve Miller Band singles chronology `` Fandango '' ( 1972 ) `` The
Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash '' ( 1974 ) `` Fandango '' ( 1972 ) `` The Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash ''
( 1974 ) `` Take the Money and Run '' / `` The Joker '' ( 1983 , double A-side , live ) Cover of the 1983 live single Steve Miller Band singles chronology ``
Living in the U.S.A. ''
</item 7>

<item 8>
Graph query row:
song: The Joker
release_date: October 1973
band: Steve Miller Band
band_relationship: CREATED
album: 7'' single (1973)
album_date: 1973
album_relationship: PART_OF
</item 8>

<item 9>
The Joker (Steve Miller Band song)
Luke Bryan often performs the song near the end of his concerts . Sampling ( edit ) `` The Joker '' was sampled in the song `` Summer and Lightning '' by
Electric Light Orchestra from their 1977 album , Out of the Blue . `` The Joker '' serves as the outline of the song `` So Cal Loco ( Party Like a Rockstar ) ''
by Sprung Monkey which appeared in the film Dude , Where 's My Car ? . It was also sampled on the original Def American Records release of the Geto Boys ' song
`` Gangster of Love '' from their 1989 album , Grip It ! On That Other Level . The sample , presumably unauthorized , was replaced with Lynyrd Skynyrd 's ``
Sweet Home Alabama '' on later pressings of the album . The song `` Hudson Dragonfly '' by Jackie - O Motherfucker , from the album , America Mystica ( 2006 ) ,
contains a sample of `` The Joker '' . The song `` Angel '' by Shaggy , from the album Hot Shot ( 2000 ) , samples the bass line of `` The Joker '' .
</item 9>

<item 10>
The Joker (Steve Miller Band song)
-- 3 : 59 Personnel ( edit ) Steve Miller -- guitar , lead vocals Gerald Johnson -- bass , backing vocals Dick Thompson -- organ John King -- drums
Certifications ( edit ) Country Certification Date Sales certified Sweden Gold February 6 , 1991 25,000 United Kingdom Silver October 1 , 1990 200,000 United
States Gold January 11 , 1974 1,000,000 Charts ( edit ) Weekly charts ( edit ) Chart ( 1973 / 1974 ) Peak position US Billboard Hot 100 Canadian RPM Top Singles
Dutch Daverende 30 18 Dutch Top 40 18 Australia Kent Music Report 8 Chart ( 1990 / 1991 ) Peak position Austrian Singles Chart 5 Dutch Nationale Top 100 Dutch
Top 40 French SNEP Singles Chart 33 Irish Singles Chart New Zealand RIANZ Singles Chart Norwegian Singles Chart Swedish Singles Chart Swiss Singles Chart 5 UK
Singles Chart Year - end charts ( edit ) Chart ( 1991 ) Position New Zealand ( Recorded Music NZ ) 20 Preceded by `` Time in a Bottle '' by Jim Croce Billboard
Hot 100 number one single January 12 ,
</item 10>
=== END ITEM id=64d616836f8a47e4 ===

=== ITEM id=c712cca992b299ec ===
Gold sentences and facts:
Sentence 0: In probability theory , two events are independent , statistically independent , or stochastically independent if the occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does not affect the probability distribution of the other .
  Fact 0.0: In probability theory, two events can be called independent.
  Fact 0.1: In probability theory, two events can be called statistically independent.
  Fact 0.2: In probability theory, two events can be called stochastically independent.
  Fact 0.3: Two events are independent if the occurrence of one does not affect the probability of occurrence of the other.
  Fact 0.4: Two random variables are independent if the realization of one does not affect the probability distribution of the other.

Retrieved items:
<item 1>
Independence (probability theory)
Using this definition , it is easy to show that if X and Y are random variables and Y is constant , then X and Y are independent , since the σ - algebra
generated by a constant random variable is the trivial σ - algebra ( ∅ , Ω ) . Probability zero events can not affect independence so independence also holds if
Y is only Pr - almost surely constant . Properties ( edit ) Self - independence ( edit ) Note that an event is independent of itself if and only if P ( A ) = P
( A ∩ A ) = P ( A ) ⋅ P ( A ) ⇔ P ( A ) = 0 or 1 ( \ displaystyle \ mathrm ( P ) ( A ) = \ mathrm ( P ) ( A \ cap A ) = \ mathrm ( P ) ( A ) \ cdot \ mathrm ( P
) ( A ) \ Leftrightarrow \ mathrm ( P ) ( A ) = 0 ( \ text ( or ) ) 1 ) . Thus an event is independent of itself if and only if it almost surely occurs or its
complement almost surely occurs .
</item 1>

<item 2>
Independence (probability theory)
More than two random variables ( edit ) A set of random variables is pairwise independent if and only if every pair of random variables is independent . Even if
the set of random variables is pairwise independent , it is not necessarily mutually independent as defined next . A set of random variables is mutually
independent if and only if for any finite subset X 1 , ... , X n ( \ displaystyle X_ ( 1 ) , \ ldots , X_ ( n ) ) and any finite sequence of numbers a 1 , ... ,
a n ( \ displaystyle a_ ( 1 ) , \ ldots , a_ ( n ) ) , the events ( X 1 ≤ a 1 ) , ... , ( X n ≤ a n ) ( \ displaystyle \ ( X_ ( 1 ) \ leq a_ ( 1 ) \ ) , \ ldots
, \ ( X_ ( n ) \ leq a_ ( n ) \ ) ) are mutually independent events ( as defined above ) . The measure - theoretically inclined may prefer to substitute events
( X ∈ A ) for events ( X ≤ a ) in the above definition , where A is any Borel set .
</item 2>

<item 3>
Independence (probability theory)
A finite set of events is mutually independent if every event is independent of any intersection of the other events -- that is , if and only if for every k -
element subset of ( A i ) i = 1 n ( \ displaystyle \ ( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) , P ( ⋂ i = 1 k A i ) = ∏ i = 1 k P ( A i ) . ( \ displaystyle \
mathrm ( P ) \ left ( \ bigcap _ ( i = 1 ) ^ ( k ) A_ ( i ) \ right ) = \ prod _ ( i = 1 ) ^ ( k ) \ mathrm ( P ) ( A_ ( i ) ) . ) This is called the
multiplication rule for independent events . Note that it is not a single condition involving only the product of all the probabilities of all single events (
see below for a counterexample ) ; it must hold true for all subsets of events .
</item 3>

<item 4>
Independence (probability theory)
1 Two random variables 1.2. 2 More than two random variables 1.2. 3 Conditional independence 1.3 Independent σ - algebras 2 Properties 2.1 Self - independence
2.2 Expectation and covariance 2.3 Characteristic function 3 Examples 3.1 Rolling dice 3.2 Drawing cards 3.3 Pairwise and mutual independence 3.4 Mutual
independence 4 See also 5 References Definition ( edit ) For events ( edit ) Two events ( edit ) Two events A and B are independent ( often written as A ⊥ B ( \
displaystyle A \ perp B ) or A ⊥ ⊥ B ( \ displaystyle A \ perp \ ! \ ! \ ! \ perp B ) ) if their joint probability equals the product of their probabilities : P
( A ∩ B ) = P ( A ) P ( B ) ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \ mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) ) .
</item 4>

<item 5>
Independence (probability theory)
) and yet no two of the three events are pairwise independent ( and hence the set of events are not mutually independent ) . This example shows that mutual
independence involves requirements on the products of probabilities of all combinations of events , not just the single events as in this example . See also (
edit ) Copula ( statistics ) Independent and identically distributed random variables Mutually exclusive events Subindependence Linear dependence between random
variables Conditional independence Normally distributed and uncorrelated does not imply independent Mean dependence References ( edit ) Jump up ^ Russell ,
Stuart ; Norvig , Peter ( 2002 ) . Artificial Intelligence : A Modern Approach . Prentice Hall . p. 478 . ISBN 0 - 13 - 790395 - 2 . ^ Jump up to : Feller , W (
1971 ) . `` Stochastic Independence '' . An Introduction to Probability Theory and Its Applications . Wiley . Jump up ^ George , Glyn , `` Testing for the
independence of three events , '' Mathematical Gazette 88 , November 2004 , 568 .
</item 5>

<item 6>
Independence (probability theory)
A ( \ displaystyle ( \ mathcal ( A ) ) ) and B ( \ displaystyle ( \ mathcal ( B ) ) ) are said to be independent if , whenever A ∈ A ( \ displaystyle A \ in ( \
mathcal ( A ) ) ) and B ∈ B ( \ displaystyle B \ in ( \ mathcal ( B ) ) ) , P ( A ∩ B ) = P ( A ) P ( B ) . ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \
mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) . )
</item 6>

<item 7>
Independence (probability theory)
Independence ( probability theory ) - wikipedia Independence ( probability theory ) Jump to : navigation , search Part of a series on Statistics Probability
theory Probability axioms Probability space Sample space Elementary event Event Random variable Probability measure Complementary event Joint probability
Marginal probability Conditional probability Independence Conditional independence Law of total probability Law of large numbers Bayes ' theorem Boole 's
inequality Venn diagram Tree diagram In probability theory , two events are independent , statistically independent , or stochastically independent if the
occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does
not affect the probability distribution of the other . The concept of independence extends to dealing with collections of more than two events or random
variables , in which case the events are pairwise independent if each pair are independent of each other , and the events are mutually independent if each event
is independent of each other combination of events . Contents ( hide ) 1 Definition 1.1 For events 1.1. 1 Two events 1.1. 2 More than two events 1.2 For random
variables 1.2. 1 Two random variables 1.2.
</item 7>

<item 8>
Independence (probability theory)
The random variables in the first space are pairwise independent because P ( A B ) = P ( A C ) = 1 / 2 = P ( A ) , P ( B A ) = P ( B C ) = 1 / 2 = P ( B ) , and
P ( C A ) = P ( C B ) = 1 / 4 = P ( C ) ; but the three random variables are not mutually independent . The random variables in the second space are both
pairwise independent and mutually independent . To illustrate the difference , consider conditioning on two events . In the pairwise independent case , although
any one event is independent of each of the other two individually , it is not independent of the intersection of the other two : P ( A B C ) = 4 40 4 40 + 1 40
= 4 5 ≠ P ( A ) ( \ displaystyle \ mathrm ( P ) ( A BC ) = ( \ frac ( \ frac ( 4 ) ( 40 ) ) ( ( \ frac ( 4 ) ( 40 ) ) + ( \ frac ( 1 ) ( 40 ) ) ) ) = ( \ tfrac
( 4 )
</item 8>

<item 9>
Independence (probability theory)
For more than two events , a mutually independent set of events is ( by definition ) pairwise independent ; but the converse is not necessarily true ( see below
for a counterexample ) . For random variables ( edit ) Two random variables ( edit ) Two random variables X and Y are independent if and only if ( iff ) the
elements of the π - system generated by them are independent ; that is to say , for every a and b , the events ( X ≤ a ) and ( Y ≤ b ) are independent events (
as defined above ) .
</item 9>

<item 10>
Independence (probability theory)
Thus , the occurrence of B does not affect the probability of A , and vice versa . Although the derived expressions may seem more intuitive , they are not the
preferred definition , as the conditional probabilities may be undefined if P ( A ) or P ( B ) are 0 . Furthermore , the preferred definition makes clear by
symmetry that when A is independent of B , B is also independent of A . More than two events ( edit ) A finite set of events ( A i ) i = 1 n ( \ displaystyle \
( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) is pairwise independent if every pair of events is independent -- that is , if and only if for all distinct pairs of
indices m , k , P ( A m ∩ A k ) = P ( A m ) P ( A k ) . ( \ displaystyle \ mathrm ( P ) ( A_ ( m ) \ cap A_ ( k ) ) = \ mathrm ( P ) ( A_ ( m ) ) \ mathrm ( P )
( A_ ( k ) ) . )
</item 10>
=== END ITEM id=c712cca992b299ec ===

=== ITEM id=f0255b89149a0469 ===
Gold sentences and facts:
Sentence 0: In the United States , there is an Animal Welfare Act that was produced in 1966 . This law protects animals in acts of research , transportation , and sale . Generally , animals are protected from any torture , neglect , or killing . There have been many amendments made towards this act to keep it updated . In order to be charged for animal abuse , there has to be prosecution and a trial . While there is only one act covering the entire United States , there are more current laws surrounding animal rights , which vary by state . Even though there are policies and severe consequences , the laws are not normally enforced with as much force or timeliness as regular crime .
  Fact 0.0: In the United States, there is an Animal Welfare Act.
  Fact 0.1: The Animal Welfare Act was produced in 1966.
  Fact 0.2: The Animal Welfare Act protects animals in acts of research, transportation, and sale.
  Fact 0.3: Under the Animal Welfare Act, animals are generally protected from any torture, neglect, or killing.
  Fact 0.4: Many amendments have been made to the Animal Welfare Act to keep it updated.
  Fact 0.5: To be charged for animal abuse, there has to be prosecution and a trial.
  Fact 0.6: Only one act covers the entire United States.
  Fact 0.7: There are more current laws surrounding animal rights, and these laws vary by state.
  Fact 0.8: There are policies and severe consequences for animal abuse.
  Fact 0.9: Animal abuse laws are not normally enforced with as much force or timeliness as regular crime.

Retrieved items:
<item 1>
Animal rights movement
It has been noted that the power of the animal rights movement in the United States is centralized in professionalized nonprofit organizations that prefer
moderated goals of welfare reform and tend to be isomorphic . The abolitionist faction diagnoses the problem as one of nonhuman animal use , offers a prognosis
including vegan outreach , and employ action frames including a rational and theoretic focus . Gary Francione , a leader in abolitionism , formed his approach
in response to the traditional movement 's focus on policy reform . Members of the abolitionist faction view policy reform as counterproductive and rely on
nonviolent education and moral suasion in their repertoire of contention . They see the promotion of veganism as a low - opportunity cost means of creating an
antispeciesist culture and posing an economic threat to the animal agriculture industrial complex . The direct action or militant faction includes in its
repertoire of contention property damage , open releases , intimidation , and direct violence . With these , they offer a prognosis to the problem of animal use
that focuses on changing society through force and fear . Animal rights actors often reject this faction , pointing to violence as a counterproductive tactic
that invites repression ( e.g.
</item 1>

<item 2>
Animal rights movement
An inclusive movement with strong group solidarity would decrease opportunity costs associated with participating ( e.g. , social stigmatization , lack of
alternatives , legal persecution ) and thus serve to increase and sustain participation in the movement . Current status of the movement ( edit ) The movement
is no longer viewed as hovering on the fringe . In the 1980s and 1990s , it was joined by a wide variety of academics and professionals , including lawyers ,
physicians , psychologists , veterinarians , and former vivisectionists , and is now a common subject of study in philosophy departments in Europe and North
America . Animal law courses are taught in 92 out of 180 law schools in the U.S. , and the movement has gained the support of senior legal scholars , including
Alan Dershowitz and Laurence Tribe of Harvard Law School . Chapters of animal rights law have been created in several state bar associations , and resolutions
related to animal rights are regularly proposed within the American Bar Association . In the 1980s , the movement became associated with punk subculture and
ideologies , particularly straight edge hardcore punk in the United States and anarcho - punk in the United Kingdom .
</item 2>

<item 3>
Animal rights movement
This association continues on into the 21st century , as evidenced by the prominence of vegan punk events such as Fluff Fest in Europe . Michael Socarras of
Greenberg Traurig told the Association of American Medical Colleges : `` There is a very important shift under way in the manner in which many people in law
schools and in the legal profession think about animals . This shift has not yet reached popular opinion . However , in ( the U.S. ) , social change has and can
occur through the courts , which in many instances do not operate as democratic institutions . Therefore , the evolution in elite legal opinion is extremely
significant ... '' Philosophical and legal aims ( edit ) Further information : Main animal rights philosophical approaches and Veganarchism The movement aims to
include animals in the moral community by putting the basic interests of non-human animals on an equal footing with the basic interests of human beings . A
basic interest would be , for example , not being made to suffer pain on behalf of other individual human or non-human animals . The aim is to remove animals
from the sphere of property and to award them personhood ; that is , to see them awarded legal rights to protect their basic interests .
</item 3>

<item 4>
Animal rights movement
, the Animal Enterprise Terrorism Act ) and does not economically or politically challenge extant systems . Smaller factions include groups focused around faith
- based animal rights theory and veganarchists , whose approach is characterized by a critique of capitalism on the grounds that it has led to mass nonhuman ,
human , and environmental exploitation . Such factionalizing , researchers have pointed out , is common to social movements and plays a role in sustaining their
health . History ( edit ) Further information : 20th century : Increase in animal use ; animal rights movement , Gary Francione , Tom Regan , Richard D. Ryder ,
and Peter Singer The Animal Rights Movement traces back to the animal protection movement in Victorian England , which was initiated by aristocratic moral
crusaders in response to the poor treatment of urban workhorses and stray dogs .
</item 4>

<item 5>
Animal rights movement
Other early influences include : Upton Sinclair 's 1906 novel The Jungle , which drew attention to obscured slaughterhouse operations ; Henry Salt 's treatises
on nonhuman animal rights ; which drew from human abolitionist arguments for recognizing personhood of people considered to be property ; and the Alcott House
of New England , a community serving as a stop on the Underground Railroad and requiring its residents to eat a vegan diet . Philosopher Peter Singer The
contemporary movement is regarded as having been founded in the UK in the early 1970s by a group of Oxford university post-graduate philosophy students , now
known as the `` Oxford Group '' . The group was led by Rosalind and Stanley Godlovitch , graduate students of philosophy who had recently become vegetarians .
The Godlovitches met John Harris and David Wood , also philosophy graduates , who were soon persuaded of the arguments in favour of animal rights and themselves
became vegetarian . The group began to actively raise the issue with pre-eminent Oxford moral philosophers , including Professor Richard Hare , both personally
and in lectures . Their approach was based not on sentimentality ( `` kindness to dumb animals ' ) , but on the moral rights of animals .
</item 5>

<item 6>
Animal rights movement
Contents ( hide ) 1 Terms and factions 2 History 3 Gender , class , and other factors 4 Current status of the movement 5 Philosophical and legal aims 6 Legal
changes influenced by the movement 7 Strategy and tactical repertoires 7.1 Use of new information communication technologies ( ICTs ) 7.2 Undercover
surveillance 7.3 Boycott 7.4 Moral shocks 7.5 Direct action 7.6 Targeting researchers 8 Criminalization of direct action methods 9 Inter-movement activity 10
Countermovement 11 Notes 12 Further reading Terms and factions ( edit ) All animal liberationists believe that the individual interests of non-human animals
deserve recognition and protection , but the movement can be split into two broad camps . Animal rights advocates , or rights liberationists , believe that
these basic interests confer moral rights of some kind on the animals , and / or ought to confer legal rights on them ; see , for example , the work of Tom
Regan .
</item 6>

<item 7>
Animal rights movement
Animal rights movement - wikipedia Animal rights movement Jump to : navigation , search This article is about the social movement . For the concept , see Animal
rights . For other uses , see Animal liberation ( disambiguation ) . The animal rights movement , sometimes called the animal liberation movement , animal
personhood , or animal advocacy movement , is a social movement which seeks an end to the rigid moral and legal distinction drawn between human and non-human
animals , an end to the status of animals as property , and an end to their use in the research , food , clothing , and entertainment industries . It is one of
the few examples of a social movement that was created , and is to a large extent sustained academically , by philosophers .
</item 7>

<item 8>
Animal rights movement
ISBN 1 - 56639 - 461 - 9 Harold D. Guither , Animal Rights : History and Scope of a Radical Social Movement ( Carbondale : Southern Illinois University Press ,
1998 ) . ISBN 0 - 8093 - 2199 - 8 James M. Jasper and Dorothy Nelkin , The Animal Rights Crusade : The Growth of a Moral Protest ( New York : The Free Press ,
1992 ) . ISBN 0 - 02 - 916195 - 9 ( hide ) Animal rights Topics Overview Abolitionism Animal protectionism Animal welfare Speciesism Vegaphobia Veganism more
...
</item 8>

<item 9>
Animal rights movement
ISSN 1187 - 7863 . doi : 10.1007 / s10806 - 015 - 9593 - 4 . ^ Jump up to : Wrenn , Corey Lee ( 2014 - 04 - 01 ) . `` Abolition Then and Now : Tactical
Comparisons Between the Human Rights Movement and the Modern Nonhuman Animal Rights Movement in the United States '' . Journal of Agricultural and Environmental
Ethics. 27 ( 2 ) : 177 -- 200 . ISSN 1187 - 7863 . doi : 10.1007 / s10806 - 013 - 9458 - 7 . ^ Jump up to : Evans , Erin ( 2010 - 07 - 01 ) . `` Constitutional
Inclusion of Animal Rights in Germany and Switzerland : How Did Animal Protection Become an Issue of National Importance ? '' . Society & Animals . 18 ( 3 ) :
231 -- 250 . ISSN 1568 - 5306 . doi : 10.1163 / 156853010X510762 . Jump up ^ `` Ethics : Animals . ''
</item 9>

<item 10>
Animal rights movement
In the United States , there is an Animal Welfare Act that was produced in 1966 . This law protects animals in acts of research , transportation , and sale .
Generally , animals are protected from any torture , neglect , or killing . There have been many amendments made towards this act to keep it updated . In order
to be charged for animal abuse , there has to be prosecution and a trial . While there is only one act covering the entire United States , there are more
current laws surrounding animal rights , which vary by state . Even though there are policies and severe consequences , the laws are not normally enforced with
as much force or timeliness as regular crime . Strategy and Tactical repertoires ( edit ) Use of New information communication technologies ( ICTs ) ( edit )
New media , such as the Internet and email , have been used by Animal Rights Movement actors and countermovement actors in a variety of capacities . Radical
factions in the movement rely on websites , blogs , podcasts , videos , and online forums to engage in vegan outreach and other mobilization efforts and build
alliances , thus overcoming exclusion by dominant factions . The use of Internet has allowed the animal rights movement to spread transnationally .
</item 10>
=== END ITEM id=f0255b89149a0469 ===

=== ITEM id=3a338c6b6853b334 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August 23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them . Nick runs over Calvin repeatedly with Travis ' truck , and the three watch in disbelief as the mutilated Calvin is still able to turn his head towards them . `` So Close , Yet So Far '' Adam Davidson Marco Ramirez August 30 , 2015 ( 2015 - 08 - 30 ) 8.18 After the run - in with zombified Calvin , Nick , Madison , and Travis choose to flee to the desert , and they want Alicia , Liza , and Chris to follow . Alicia finds Matt ill in his disheveled house . Travis arrives and sees a bite on Matt , who convinces Alicia to leave . The group returns to Madison 's home to gather supplies . Nick suffers from heroin withdrawal , so Madison drives to her school to get him Oxycodone . There , she encounters Tobias scavenging food . A zombified Artie tries to bite Tobias , so Madison kills Artie and drives Tobias home . Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside , a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not before Griselda is injured by a collapsing scaffold . Unable to reach a hospital , the group drives to Madison 's house , where Nick , Madison , and Alicia temporarily flee when the zombified Mr. Dawson attempts to enter , attracted by the barking dog Nick had let in . Nick leads Madison and Alicia to the Trans ' house next door , where they take a shotgun . Travis arrives and is attacked by Mr. Dawson , who is shot and killed by Daniel . All three families decide to stay the night and evacuate in the morning . Nurse Liza tends to Griselda 's injured foot but notes that Griselda will die if not treated by a doctor . Ofelia tells Daniel they should flee with Travis , but Daniel insists his family can survive alone and will join his cousin later . The next morning , as the Clarks and Manawas start driving away , the National Guard arrives and quarantines the block . While Travis says , `` It 's going to get better , '' Daniel laments that it 's , `` too late , '' as he watches a guardsman mark the neighboring house . `` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda 's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not technically a nurse . Daniel warns Madison of his El Salvador experience , when the sick were taken under the guise of receiving hospitalization but instead killed . Soldiers take Griselda and Nick to a hospital , but Nick 's family protests his departure . Liza agrees to go to assist the medical team , despite not wanting to leave her son . Travis retreats to the roof and sees the signal from the Dead Zone . Seconds later , he sees and hears gunfire , followed by darkness . 5 5 `` Cobalt '' Kari Skogland David Wiener September 27 , 2015 ( 2015 - 09 - 27 ) 6.66 In a military cell , Strand bribes a guardsman to save feverish Nick from being moved . Liza helps Dr. Exner with patients at the hospital . Chris is devastated that Liza left voluntarily to help at the hospital , but Travis promises to bring her back . Madison discovers Daniel detaining Adams in the Trans ' basement . Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside . 6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile , the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick , Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis brutally beats Adams and leaves him for dead . Strand leads the group to his oceanside mansion , where he reveals to Nick that he owns a yacht which he plans to escape on , called the Abigail . On the beach , Liza reveals to Madison that she had been bitten during the escape . Liza pleads with Madison and Travis to euthanize her before she turns . Travis promises to protect Chris before shooting Liza .
  Fact 0.0: The table lists each episode's overall number, season number, title, director, writer, original air date, and U.S. viewers in millions.
  Fact 0.1: Episode 'Pilot' was directed by Adam Davidson and written by Robert Kirkman and Dave Erickson.
  Fact 0.2: Episode 'Pilot' originally aired on August 23, 2015.
  Fact 0.3: Episode 'Pilot' had 10.13 million U.S. viewers.
  Fact 0.4: Nick awakens in a heroin den in an abandoned church, to find his girlfriend Gloria eating a corpse.
  Fact 0.5: While fleeing, Nick is hit by a car and hospitalized.
  Fact 0.6: The doctor tells Madison and Travis that Nick's claims about the incident are heroin hallucinations.
  Fact 0.7: Travis believes Nick after Travis visits the church himself.
  Fact 0.8: Alicia becomes more worried about Nick's chemical dependency.
  Fact 0.9: The next day, school closes early due to high levels of absenteeism and rumors of an epidemic.
  Fact 0.10: Nick escapes from the hospital and meets with Calvin, hoping to learn if the drugs Calvin sold him caused him to hallucinate in the church.
  Fact 0.11: Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs.
  Fact 0.12: In the ensuing struggle, Calvin is mortally shot.
  Fact 0.13: After Travis and Madison arrive, the zombified Calvin attacks them.
  Fact 0.14: Nick runs over Calvin repeatedly with Travis's truck.
  Fact 0.15: The three watch in disbelief as the mutilated Calvin is still able to turn his head towards them.
  Fact 0.16: Episode 'So Close, Yet So Far' was directed by Adam Davidson and written by Marco Ramirez.
  Fact 0.17: Episode 'So Close, Yet So Far' originally aired on August 30, 2015.
  Fact 0.18: Episode 'So Close, Yet So Far' had 8.18 million U.S. viewers.
  Fact 0.19: After the run-in with zombified Calvin, Nick, Madison, and Travis choose to flee to the desert.
  Fact 0.20: Nick, Madison, and Travis want Alicia, Liza, and Chris to follow them to the desert.
  Fact 0.21: Alicia finds Matt ill in his disheveled house.
  Fact 0.22: Travis arrives and sees a bite on Matt.
  Fact 0.23: Matt convinces Alicia to leave.
  Fact 0.24: The group returns to Madison's home to gather supplies.
  Fact 0.25: Nick suffers from heroin withdrawal, so Madison drives to her school to get him Oxycodone.
  Fact 0.26: At the school, Madison encounters Tobias scavenging food.
  Fact 0.27: A zombified Artie tries to bite Tobias, so Madison kills Artie and drives Tobias home.
  Fact 0.28: Chris's bus is trapped in a traffic jam caused by a zombie shooting by the police.
  Fact 0.29: Chris films the event and joins in a protest against that and other recent fatal police shootings, when Travis and Liza meet up with him.
  Fact 0.30: A riot erupts after police shoot down another zombie.
  Fact 0.31: The three Manawas find refuge with the Salazars in their gated barbershop-home.
  Fact 0.32: Travis tells Madison to take the kids to the desert without him, and he will catch up.
  Fact 0.33: Alicia witnesses her zombified neighbor, Mr. Dawson, attacking Mrs. Cruz across the street.
  Fact 0.34: Madison prevents Alicia from intervening.
  Fact 0.35: The group inside the barbershop remains trapped, while the riot outside intensifies.
  Fact 0.36: Episode 'The Dog' was directed by Adam Davidson and written by Jack LoGiudice.
  Fact 0.37: Episode 'The Dog' originally aired on September 13, 2015.
  Fact 0.38: Episode 'The Dog' had 7.19 million U.S. viewers.
  Fact 0.39: While a riot rages outside, a mob sets fire to the store adjoining the barbershop, forcing the Salazars and Manawas to flee.
  Fact 0.40: The group reaches Travis's truck and escapes, but not before Griselda is injured by a collapsing scaffold.
  Fact 0.41: Unable to reach a hospital, the group drives to Madison's house.
  Fact 0.42: Nick, Madison, and Alicia temporarily flee when the zombified Mr. Dawson attempts to enter.
  Fact 0.43: Mr. Dawson is attracted by the barking dog Nick had let in.
  Fact 0.44: Nick leads Madison and Alicia to the Trans' house next door, where they take a shotgun.
  Fact 0.45: Travis arrives and is attacked by Mr. Dawson.
  Fact 0.46: Mr. Dawson is shot and killed by Daniel.
  Fact 0.47: All three families decide to stay the night and evacuate in the morning.
  Fact 0.48: Nurse Liza tends to Griselda's injured foot but notes that Griselda will die if not treated by a doctor.
  Fact 0.49: Ofelia tells Daniel they should flee with Travis.
  Fact 0.50: Daniel insists his family can survive alone and will join his cousin later.
  Fact 0.51: The next morning, as the Clarks and Manawas start driving away, the National Guard arrives and quarantines the block.
  Fact 0.52: Travis says 'It's going to get better.'
  Fact 0.53: Daniel laments that it's 'too late' as he watches a guardsman mark the neighboring house.
  Fact 0.54: Episode 'Not Fade Away' was directed by Kari Skogland and written by Meaghan Oppenheimer.
  Fact 0.55: Episode 'Not Fade Away' originally aired on September 20, 2015.
  Fact 0.56: Episode 'Not Fade Away' had 6.62 million U.S. viewers.
  Fact 0.57: Days after the National Guard quarantines the neighborhood into a Safe Zone, residents try to live normally.
  Fact 0.58: Tensions build under the military rule.
  Fact 0.59: Madison is stressed by extra work caused by her home's being overcrowded.
  Fact 0.60: Madison is stressed by Travis's new role as civilian liaison.
  Fact 0.61: Chris shows a video to Travis and Madison of a light signaling from the Dead Zone.
  Fact 0.62: Travis talks Doug into getting mental help.
  Fact 0.63: Liza medically helps neighbors.
  Fact 0.64: Nick steals morphine from Hector via IV drip.
  Fact 0.65: Ofelia kisses Adams, who was unable to get Griselda's medicine.
  Fact 0.66: Madison exchanges signals with the flashing light in the Dead Zone.
  Fact 0.67: Madison sneaks outside the fence to investigate.
  Fact 0.68: Madison finds evidence that the guardsmen killed civilians, even the uninfected.
  Fact 0.69: Travis learns that Doug has been hospitalized for his mental issues.
  Fact 0.70: Dr. Exner determines that Liza is not technically a nurse.
  Fact 0.71: Daniel warns Madison of his El Salvador experience, when the sick were taken under the guise of receiving hospitalization but instead killed.
  Fact 0.72: Soldiers take Griselda and Nick to a hospital, but Nick's family protests his departure.
  Fact 0.73: Liza agrees to go to assist the medical team, despite not wanting to leave her son.
  Fact 0.74: Travis retreats to the roof and sees the signal from the Dead Zone.
  Fact 0.75: Seconds later, Travis sees and hears gunfire, followed by darkness.
  Fact 0.76: Episode 'Cobalt' is overall number 5 and season number 5.
  Fact 0.77: Episode 'Cobalt' was directed by Kari Skogland and written by David Wiener.
  Fact 0.78: Episode 'Cobalt' originally aired on September 27, 2015.
  Fact 0.79: Episode 'Cobalt' had 6.66 million U.S. viewers.
  Fact 0.80: In a military cell, Strand bribes a guardsman to save feverish Nick from being moved.
  Fact 0.81: Liza helps Dr. Exner with patients at the hospital.
  Fact 0.82: Chris is devastated that Liza left voluntarily to help at the hospital.
  Fact 0.83: Travis promises to bring Liza back.
  Fact 0.84: Madison discovers Daniel detaining Adams in the Trans' basement.
  Fact 0.85: Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family.
  Fact 0.86: Strand recruits Nick for an escape plot.
  Fact 0.87: Travis convinces Moyers' squad to take him to the hospital to check on his friends.
  Fact 0.88: While en route, Moyers encourages Travis to shoot a zombie.
  Fact 0.89: Travis is emotionally unable to pull the trigger.
  Fact 0.90: The soldiers stop to assist another squad in a building infested by zombies.
  Fact 0.91: Most of those soldiers, including Moyers, are overcome.
  Fact 0.92: The few survivors flee and drop off Travis near the Safe Zone.
  Fact 0.93: Travis learns that Daniel tortured Adams into revealing what 'Cobalt' means.
  Fact 0.94: 'Cobalt' means that in the morning, all civilians will be killed, and the guardsmen will evacuate the city.
  Fact 0.95: Griselda dies of septic shock at the hospital.
  Fact 0.96: Liza shoots Griselda's brain to prevent reanimation.
  Fact 0.97: Daniel visits a nearby sports arena to verify Adams' story that it was sealed with 2,000 now-zombified civilians inside.
  Fact 0.98: Episode 'The Good Man' is overall number 6 and season number 6.
  Fact 0.99: Episode 'The Good Man' was directed by Stefan Schwartz and written by Robert Kirkman and Dave Erickson.
  Fact 0.100: Episode 'The Good Man' originally aired on October 4, 2015.
  Fact 0.101: Episode 'The Good Man' had 6.86 million U.S. viewers.
  Fact 0.102: The group drives to the National Guard's headquarters to rescue Liza, Griselda, and Nick.
  Fact 0.103: Adams agrees to be their guide when let go by Travis.
  Fact 0.104: The group infiltrates the base after Daniel distracts the guards by leading a horde of walkers from the arena.
  Fact 0.105: Travis, Madison, Daniel, and Ofelia go inside, while Alicia and Chris stay behind.
  Fact 0.106: Meanwhile, the walkers breach the perimeter defenses and swarm the base.
  Fact 0.107: Travis's group reaches the holding cells and sets the detainees free.
  Fact 0.108: Travis's group reunites with Nick, Liza, and Strand before trying to escape.
  Fact 0.109: The group tries to escape through the medical ward.
  Fact 0.110: The group discovers Dr. Exner has euthanized all of the patients in the medical ward.
  Fact 0.111: Dr. Exner tells them of an escape route before presumably committing suicide.
  Fact 0.112: Before they can escape, the group encounters Adams, who shoots Ofelia in the arm.
  Fact 0.113: Enraged, Travis brutally beats Adams and leaves him for dead.
  Fact 0.114: Strand leads the group to his oceanside mansion.
  Fact 0.115: At the mansion, Strand reveals to Nick that he owns a yacht which he plans to escape on, called the Abigail.
  Fact 0.116: On the beach, Liza reveals to Madison that she had been bitten during the escape.
  Fact 0.117: Liza pleads with Madison and Travis to euthanize her before she turns.
  Fact 0.118: Travis promises to protect Chris before shooting Liza.

Retrieved items:
<item 1>
Fear the Walking Dead (season 1)
TV by the Numbers . October 2 , 2015 . Retrieved March 27 , 2018 . Jump up ^ Lambert , David ( October 6 , 2015 ) . `` Fear the Walking Dead - Press Release
Announces ' The Complete 1st Season ' on Blu - ray , DVD '' . TVShowsOnDVD.com . Retrieved October 6 , 2015 . Jump up ^ Lambert , David ( January 20 , 2016 ) .
`` Fear the Walking Dead - ' The Complete 1st Season : Special Edition 's ' Press Release ! '' . TVShowsOnDVD.com . Retrieved January 21 , 2016 . External links
( edit ) Official website List of Fear the Walking Dead episodes on IMDb ( hide ) The Walking Dead Robert Kirkman Tony Moore Charlie Adlard Comic books The
Walking Dead Television series The Walking Dead Episodes season 1 5 6 7 8 Accolades Soundtrack Vol.
</item 1>

<item 2>
Fear the Walking Dead (season 1)
Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent
fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with
the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her
zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains
trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside
, a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not
before Griselda is injured by a collapsing scaffold .
</item 2>

<item 3>
Fear the Walking Dead (season 1)
Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad
to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the
trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few
survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all
civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent
reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside .
</item 3>

<item 4>
Fear the Walking Dead (season 1)
Fear the Walking Dead ( season 1 ) - wikipedia Fear the Walking Dead ( season 1 ) Jump to : navigation , search Fear the Walking Dead ( season 1 ) Home media
cover art Starring Kim Dickens Cliff Curtis Frank Dillane Alycia Debnam - Carey Elizabeth Rodriguez Mercedes Mason Lorenzo James Henrie Rubén Blades Country of
origin United States No. of episodes 6 Release Original network AMC Original release August 23 ( 2015 - 08 - 23 ) -- October 4 , 2015 ( 2015 - 10 - 04 ) Season
chronology Next → Season 2 List of Fear the Walking Dead episodes The first season of Fear the Walking Dead , an American horror - drama television series on
AMC , premiered on August 23 , 2015 , and concluded on October 4 , 2015 , consisting of six episodes . The series is a companion series and prequel to The
Walking Dead , which is based on the comic book series of the same name by Robert Kirkman , Tony Moore , and Charlie Adlard .
</item 4>

<item 5>
Fear the Walking Dead (season 1)
Rotten Tomatoes . Retrieved August 25 , 2015 . Jump up ^ `` Fear the Walking Dead : Season 1 '' . Metacritic . Retrieved August 25 , 2015 . Jump up ^
Vincentelli , Elisabeth ( August 20 , 2015 ) . `` ' Walking Dead ' prequel is creepily suspenseful '' . New York Post . Retrieved August 20 , 2015 . Jump up ^
Tucker , Ken ( August 5 , 2015 ) . `` ' Fear the Walking Dead ' : Artful Suspense , Not Bloody Horror '' . Yahoo TV . Retrieved August 20 , 2015 . Jump up ^
Goodman , Tim ( August 4 , 2015 ) . `` ' Fear the Walking Dead ' : TV Review '' . The Hollywood Reporter . Retrieved August 20 , 2015 . Jump up ^ Fienberg ,
Daniel ( August 20 , 2015 ) . `` No. 296 - ' Fear the Walking Dead ' and more '' . HitFix .
</item 5>

<item 6>
Fear the Walking Dead (season 1)
1 Talking Dead Fear the Walking Dead Episodes season 1 Web series The Walking Dead Torn Apart Cold Storage Fear the Walking Dead Flight 462 Miscellaneous Video
games Season One `` 400 Days '' Two Michonne Three : A New Frontier Other games Survival Instinct Road to Survival Rides The Walking Dead : The Ride Novels Rise
of the Governor The Road to Woodbury The Fall of the Governor Descent Invasion Search and Destroy Return to Woodbury Characters Lists Comic book series
Television series Aaron Alden Amy Jessie Anderson Andrea Axel Tara Chambler Clementine Daryl Dixon Merle Dixon Dwight Rosita Espinosa Lee Everett King Ezekiel
Abraham Ford Gareth The Governor Gregory Beth Greene Hershel Greene Maggie Greene Carl Grimes Lori Grimes Rick Grimes Dale Horvath Jadis Morgan Jones Lilly
Milton Mamet Caesar Martinez Deanna Monroe Paul `` Jesus '' Monroe Spencer Monroe Morales Michonne Negan Olivia The Prisoners Carol Peletier Sophia Peletier
Eugene Porter Glenn Rhee Lizzie and Mika Samuels Simon Gabriel Stokes Bob Stookey T - Dog Shane Walsh Sasha Williams Tyreese Williams Retrieved from ``
https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Fear the Walking Dead (season 1)
overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August
23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is
hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick
after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of
absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to
hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the
ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them .
</item 7>

<item 8>
Fear the Walking Dead (season 1)
6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's
headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts
the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile ,
the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick ,
Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an
escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis
brutally beats Adams and leaves him for dead .
</item 8>

<item 9>
Fear the Walking Dead (season 1)
Dr. Exner determines that Liza is not technically a nurse . Daniel warns Madison of his El Salvador experience , when the sick were taken under the guise of
receiving hospitalization but instead killed . Soldiers take Griselda and Nick to a hospital , but Nick 's family protests his departure . Liza agrees to go to
assist the medical team , despite not wanting to leave her son . Travis retreats to the roof and sees the signal from the Dead Zone . Seconds later , he sees
and hears gunfire , followed by darkness . 5 5 `` Cobalt '' Kari Skogland David Wiener September 27 , 2015 ( 2015 - 09 - 27 ) 6.66 In a military cell , Strand
bribes a guardsman to save feverish Nick from being moved . Liza helps Dr. Exner with patients at the hospital . Chris is devastated that Liza left voluntarily
to help at the hospital , but Travis promises to bring her back . Madison discovers Daniel detaining Adams in the Trans ' basement . Alicia and Chris get drunk
in and vandalize the abandoned home of a wealthy family .
</item 9>

<item 10>
Fear the Walking Dead (season 1)
`` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood
into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being
overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug
into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda
's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the
guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not
technically a nurse .
</item 10>
=== END ITEM id=3a338c6b6853b334 ===

=== ITEM id=71273e3d5ae9d5e1 ===
Gold sentences and facts:
Sentence 0: Somatic motor neurons originate in the central nervous system , project their axons to skeletal muscles ( such as the muscles of the limbs , abdominal , and intercostal muscles ) , which are involved in locomotion . The three types of these neurons are the alpha efferent neurons , beta efferent neurons , and gamma efferent neurons . They are called efferent to indicate the flow of information from the central nervous system ( CNS ) to the periphery .
  Fact 0.0: Somatic motor neurons originate in the central nervous system.
  Fact 0.1: Somatic motor neurons project their axons to skeletal muscles.
  Fact 0.2: Examples of those skeletal muscles are the muscles of the limbs, abdominal muscles, and intercostal muscles.
  Fact 0.3: The skeletal muscles that somatic motor neurons project to are involved in locomotion.
  Fact 0.4: There are three types of somatic motor neurons.
  Fact 0.5: The three types are alpha efferent neurons, beta efferent neurons, and gamma efferent neurons.
  Fact 0.6: Somatic motor neurons are called efferent.
  Fact 0.7: The name efferent indicates the flow of information from the central nervous system (CNS) to the periphery.
Sentence 1: The term ' motor neuron ' is usually restricted to the lower motor neurons , the efferent nerves that directly innervate muscles .
  Fact 1.0: The term 'motor neuron' is usually restricted to the lower motor neurons.
  Fact 1.1: Lower motor neurons are the efferent nerves that directly innervate muscles.

Retrieved items:
<item 1>
Motor neuron
These columns are further explained in the table below . Motor Columns of Spinal Cord Motor Column Location in Spinal Cord Target Median Motor Column Present
entire length Axial muscles Hypaxial Motor Column Thoracic Region Body wall muscles Preganglionic Motor Column Thoracic Region Sympathetic ganglion Laterl Motor
Column Brachial and Lumbar region ( both regions are further divided into medial and lateral domains ) Muscles of the limbs Phrenic Motor Coulmn Cervical Region
Diaphragm Anatomy and Physiology ( edit ) Spinal cord tracts Location of lower motor neurons in spinal cord Upper motor neurons ( edit ) Upper motor neurons
originate in the motor cortex located in the precentral gyrus . The cells that make up the primary motor cortex are Betz cells , which are a type of pyramidal
cell . The axons of these cells descend from the cortex to form the corticospinal tract . White matter Tract ( edit ) White matter tracts are bundles of axons
and serve as means to propagate nerve impulses on a large scale . These are found in the spinal cord and appear white from the myelin surrounding axons .
</item 1>

<item 2>
Motor neuron
Branch of NS Position Neurotransmitter Somatic n / a Acetylcholine Parasympathetic Preganglionic Acetylcholine Parasympathetic Ganglionic Acetylcholine
Sympathetic Preganglionic Acetylcholine Sympathetic Ganglionic Norepinephrine * * Except fibers to sweat glands and certain blood vessels Motor neuron
neurotransmitters General visceral motor neurons ( edit ) These motor neurons indirectly innervate cardiac muscle and smooth muscles of the viscera ( the
muscles of the arteries ) : they synapse onto neurons located in ganglia of the autonomic nervous system ( sympathetic and parasympathetic ) , located in the
peripheral nervous system ( PNS ) , which themselves directly innervate visceral muscles ( and also some gland cells ) . In consequence : the motor command of
skeletal and branchial muscles is monosynaptic ( involving only one motor neuron , respectively , somatic and branchial , which synapses onto the muscle ) .
</item 2>

<item 3>
Motor neuron
There are two types of motor neuron -- upper motor neurons and lower motor neurons . Axons from upper motor neurons synapse onto interneurons in the spinal cord
and occasionally directly onto lower motor neurons . The axons from the lower motor neurons are efferent nerve fibers that carry signals from the spinal cord to
the effectors . Types of lower motor neurons are alpha motor neurons , beta motor neurons , and gamma motor neurons . The term ' motor neuron ' is usually
restricted to the lower motor neurons , the efferent nerves that directly innervate muscles . A single motor neuron may innervate many muscle fibres and a
muscle fibre can undergo many action potentials in the time taken for a single muscle twitch . As a result , if an action potential arrives before a twitch has
completed , the twitches can superimpose on one another , either through summation or a tetanic contraction . In summation , the muscle is stimulated
repetitively such that additional action potentials coming from the somatic nervous system arrive before the end of the twitch . The twitches thus superimpose
on one another , leading to a force greater than that of a single twitch .
</item 3>

<item 4>
Motor neuron
There are two types of gamma motor neurons : Dynamic - These focus on Bag1 fibers and enhance dynamic sensitivity . Static - These focus on Bag2 fibers and
enhance stretch sensitivity . Regulatory factors of lower motor Neurons Size Principle -- this relates to the of the soma of the motor neuron . This restricts
larger neurons to receive a larger excitatory signal in order to stimulate the muscle fibers it innervates . By reducing unnecessary muscle fiber recruitment ,
the body is able to optimize energy consumption . Persistent Inward Current ( PIC ) -- recent animal study research has shown that constant flow of ions such as
calcium and sodium through channels in the soma and dendrites influence the synaptic input . An alternate way to think of this is that the post-synaptic neuron
is being primed before receiving an impulse . After Hyper - polarization ( AHP ) -- we are all familiar with the idea of hyperpolarization after an action
potential . A trend has been identified that shows slow motor neurons to have more intense AHPs for a longer duration . One way to remember this is that slow
muscle fibers can contract for loner , so it makes since that their corresponding motor neurons fire at a slower rate .
</item 4>

<item 5>
Motor neuron
Motor neuron - wikipedia Motor neuron Jump to : navigation , search Motor neuron Micrograph of the hypoglossal nucleus showing motor neurons with their
characteristic coarse Nissl substance ( `` tigroid '' cytoplasm ) . H&E - LFB stain . Details Location Ventral horn of the spinal cord , some cranial nerve
nuclei Shape Projection neuron Function Excitatory projection ( to NMJ ) Neurotransmitter UMN to LMN : glutamate ; LMN to NMJ : ACh Presynaptic connections
Primary motor cortex via the Corticospinal tract Postsynaptic connections Muscle fibers and other neurons Identifiers MeSH D009046 NeuroLex ID nifext_103 TA
A14. 2.00. 021 FMA 5867 Anatomical terms of neuroanatomy ( edit on Wikidata ) A motor neuron ( or motoneuron ) is a neuron whose cell body is located in the
motor cortex , brainstem or the spinal cord , and whose axon ( fiber ) projects to the spinal cord or outside of the spinal cord to directly or indirectly
control effector organs , mainly muscles and glands . There are two types of motor neuron -- upper motor neurons and lower motor neurons .
</item 5>

<item 6>
Motor neuron
the command of visceral muscles is disynaptic ( involving two neurons : the general visceral motor neuron located in the CNS , which synapses onto a ganglionic
neuron , located in the PNS , which synapses onto the muscle ) . All vertebrate motor neurons are cholinergic , that is , they release the neurotransmitter
acetylcholine . Parasympathetic ganglionic neurons are also cholinergic , whereas most sympathetic ganglionic neurons are noradrenergic , that is , they release
the neurotransmitter noradrenaline . ( see Table ) Motor neurons in Regenerative medicine ( edit ) Human lower motor neurons can be generated in vitro from
embryonic stem cells and induced pluripotent stem cells . See also ( edit ) Betz cell Central chromatolysis Motor dysfunction Nerve References ( edit ) Jump up
^ Tortora , Gerard ; Derrickson , Bryan ( 2014 ) . Principles of Anatomy & Physiology ( 14th ed . ) . New Jersey : John Wiley & Sons , Inc . pp. 406 , 502 , 541
. ISBN 978 - 1 - 118 - 34500 - 9 .
</item 6>

<item 7>
Motor neuron
Muscle relaxation and inhibition of muscle contraction in vertebrates is obtained only by inhibition of the motor neuron itself . Muscle innervation may
eventually play a role in the maturation of motor activity . This is how muscle relaxants work by acting on the motor neurons that innervate muscles ( by
decreasing their electrophysiological activity ) or on cholinergic neuromuscular junctions , rather than on the muscles themselves . Special visceral motor
neurons ( edit ) These are also known as branchial motor neurons , which are involved in facial expression , mastication , phonation , and swallowing .
Associated cranial nerves are the oculomotor , abducens , trochlear , and hypoglossal nerves .
</item 7>

<item 8>
Motor neuron
There are three primary categories lower motor neurons , which can be further divided in sub-categories . According to their targets , motor neurons are
classified into three broad categories : Somatic motor neurons Special visceral motor neurons General visceral motor neurons Somatic motor neurons ( edit )
Somatic motor neurons originate in the central nervous system , project their axons to skeletal muscles ( such as the muscles of the limbs , abdominal , and
intercostal muscles ) , which are involved in locomotion . The three types of these neurons are the alpha efferent neurons , beta efferent neurons , and gamma
efferent neurons . They are called efferent to indicate the flow of information from the central nervous system ( CNS ) to the periphery . Alpha motor neurons
innervate extrafusal muscle fibers , which are the main force - generating component of a muscle . Their cell bodies are in the ventral horn of the spinal cord
and they are sometimes called ventral horn cells . A single motor neuron may synapse with 150 muscle fibers on average . The motor neuron and all of the muscle
fibers to which it connects is a motor unit .
</item 8>

<item 9>
Motor neuron
A tetanic contraction is caused by constant , very high frequency stimulation - the action potentials come at such a rapid rate that individual twitches are
indistinguishable , and tension rises smoothly eventually reaching a plateau . Contents ( hide ) 1 Development 2 Anatomy and physiology 2.1 Upper motor neurons
2.2 White Matter Tract 2.3 Lower Motor Neurons 2.4 Somatic motor neurons 2.4. 1 Neuromuscular Junction 2.5 Special visceral motor neurons 2.6 General visceral
motor neurons 3 Motor neurons in regenerative medicine 4 See also 5 References 6 Sources Development ( edit ) Motor neurons begin to develop early in embryonic
development , and motor function continues to develop well into childhood . In the neural tube cells are specified to either the rostral - caudal axis or
ventral - dorsal axis . The axons of motor neurons begin to appear in the fourth week of development from the ventral region of the ventral - dorsal axis ( the
basal plate ) . This homeodomain is known as the motor neural progenitor domain ( pMN ) .
</item 9>

<item 10>
Motor neuron
These use both oxidative and glycolytic means to gain energy . In addition to voluntary skeletal muscle contraction , alpha motor neurons also contribute to
muscle tone , the continuous force generated by noncontracting muscle to oppose stretching . When a muscle is stretched , sensory neurons within the muscle
spindle detect the degree of stretch and send a signal to the CNS . The CNS activates alpha motor neurons in the spinal cord , which cause extrafusal muscle
fibers to contract and thereby resist further stretching . This process is also called the stretch reflex . Beta motor neurons innervate intrafusal muscle
fibers of muscle spindles , with collaterals to extrafusal fibres . There are two types of beta motor neurons : Slow Contracting - These innervate extrafusal
fibers . Fast Contracting - These innervate intrafusal fibers . Gamma motor neurons innervate intrafusal muscle fibers found within the muscle spindle . They
regulate the sensitivity of the spindle to muscle stretching . With activation of gamma neurons , intrafusal muscle fibers contract so that only a small stretch
is required to activate spindle sensory neurons and the stretch reflex .
</item 10>
=== END ITEM id=71273e3d5ae9d5e1 ===

=== ITEM id=633cad57cf12417f ===
Gold sentences and facts:
Sentence 0: The Pan-American Highway is a network of roads measuring about 30,000 kilometres ( 19,000 mi ) in total length . Except for a rainforest break of approximately 160 km ( 100 mi ) , called the Darién Gap , the road links almost all of the Pacific coastal countries of the Americas in a connected highway system . According to Guinness World Records , the Pan-American Highway is the world 's longest `` motorable road '' . However , because of the Darién Gap , it is not possible to cross between South America and Central America with conventional highway vehicles . Without an all - terrain vehicle , it is necessary to circumnavigate this terrestrial stretch by sea .
  Fact 0.0: The Pan-American Highway is a network of roads.
  Fact 0.1: The Pan-American Highway measures about 30,000 kilometres (19,000 mi) in total length.
  Fact 0.2: The Pan-American Highway has a rainforest break of approximately 160 km (100 mi).
  Fact 0.3: The rainforest break of the Pan-American Highway is called the Darién Gap.
  Fact 0.4: Apart from the Darién Gap, the Pan-American Highway links almost all of the Pacific coastal countries of the Americas.
  Fact 0.5: The Pan-American Highway links those countries in a connected highway system.
  Fact 0.6: According to Guinness World Records, the Pan-American Highway is the world's longest 'motorable road'.
  Fact 0.7: Because of the Darién Gap, it is not possible to cross between South America and Central America with conventional highway vehicles.
  Fact 0.8: Without an all-terrain vehicle, it is necessary to go around the terrestrial stretch of the Darién Gap by sea.

Retrieved items:
<item 1>
Pan-American Highway
Contents ( hide ) 1 Development and construction 2 Countries served 3 Hemispheric overview 4 Northern section 4.1 Alaska , Canada and the contiguous 48 states
of the United States 4.2 Mexico 4.3 Central America 5 Darién Gap 6 Southern section 6.1 Colombia and Venezuela 6.2 Ecuador , Peru , Chile , Argentina and
Paraguay 6.3 Brazil and Uruguay 6.4 Guyana , Suriname and French Guyana 7 West Indies section 8 Art and culture 9 Photo gallery 10 See also 11 References 12
Sources 13 External links Development and construction ( edit ) The concept of an overland route from one tip of the Americas to the other was originally
proposed at the First Pan-American Conference in 1889 as a railroad ; however , this proposal was never realized . The idea of building a highway emerged at the
Fifth International Conference of American States in 1923 . The first conference regarding construction of the highway occurred on October 5 , 1925 .
</item 1>

<item 2>
Pan-American Highway
Panamericana -- Pan American Highway -- in Pichincha , Ecuador , near Cashapamba Panamericana -- northern Peru near Pacasmayo Panamericana near Puerto De Lomas
, Peru Panamericana in the Atacama Desert northern Chile GMC Suburban traveled all of the Pan-American Highway . Patagonia , Chile . Panamericana - Pan American
Highway - in the Cordillera de los Andes , southern Ecuador , near to Catacocha , 2500m / 8000 ft Elev . National Route 7 -- Pan-American Highway in Junín ,
Buenos Aires , Argentina See also ( edit ) CANAMEX Corridor Continental 1 Trans - Siberian Highway Buenos Aires References ( edit ) Jump up ^ French : Route /
Autoroute Panaméricaine / Transaméricaine , Portuguese : Rodovia / Auto - estrada Pan-americana , Spanish : Autopista / Carretera / Ruta Panamericana Jump up ^
`` A Gap in the Andes : Image of the Day '' . earthobservatory.nasa.gov. 2015 - 04 - 02 .
</item 2>

<item 3>
The Pan-American Highway is a roughly 30,000-kilometre (19,000-mile) network linking nearly all Pacific-coast countries of the Americas. Its main route is
interrupted by the approximately 160-kilometre (100-mile) Darién Gap, a rainforest region without a road connection. Guinness World Records calls it the world’s
longest motorable road. The route has featured in travel writing, including Tim Cahill’s record-setting drive from Argentina to Alaska, and
</item 3>

<item 4>
Pan-American Highway
In 2003 , Kevin Sanders , a long - distance rider , broke the Guinness World Record for the fastest traversal of the highway by motorcycle in 34 days . Photo
gallery ( edit ) The northern end of the Pan-American Highway at Deadhorse , Alaska Interstate 25 ( Pan-American Freeway ) approaching the Big I interchange in
Albuquerque , New Mexico , USA Pan-American Highway through San Martin , El Salvador . Another view of the Pan-American Highway in El Salvador . Pan-American
Highway in El Salvador between Lourdes and Santa Ana ; this flat 1.5 km long straight section can be used as an airstrip and it was used during El Salvador
Civil War . Highway in Guanacaste , Costa Rica ( going towards the Nicaraguan border , still many kilometres away . ) Pan-American Highway in Tres Rios , Costa
Rica , right before the toll plaza ( about 337 more km to go until the Panamanian border ) .
</item 4>

<item 5>
The Pan-American Highway has disputed southern termini, including Puerto Montt, Quellón, and Ushuaia, and is interrupted by the Darién Gap. Through Mexico and
Central America it is called the Inter-American Highway; Mexico’s main route follows Federal Highway 85 from Nuevo Laredo to Mexico City, with border spurs
including Nogales, Ciudad Juárez, and others. Since 1966, the entire US Interstate system has been designated part of the Pan
</item 5>

<item 6>
The Pan-American Highway was first proposed as a railroad in 1889; a highway plan emerged in 1923, with construction discussions beginning in 1925. In 1937, 14
countries signed an agreement to build it, and Mexico completed its section in 1950. The mostly complete system extends from Prudhoe Bay, Alaska, to southern
South America. Its northern route serves nine countries and its southern route eight, with spurs into Bolivia, Brazil, Paraguay, and
</item 6>

<item 7>
Pan-American Highway
They barely succeeded in thrashing a passage through the extreme terrain . In 1979 a team led by Mark Smith drove standard production CJ7 - model Jeeps from
South to North , traversing the Gap - with difficulty . In June 1984 , Loren and Patty Upton took 741 days slogging , winching , chopping and digging their way
through the inhospitable jungles of the Darién Gap . One proposed option to bridge the gap is a short ferry link from Colombia to a new ferry port in Panama ,
with an extension of the existing Panama highway that would complete the highway without violating these environmental concerns . Southern section ( edit ) Main
article : Pan-American Highway ( South America ) A Vía PanAm shield sign is sometimes found on routes in South American countries ( such as Chile ) associated
with the Pan-American Highway . Sculpture of a native man standing at the entrance of Fusagasugá , Colombia , over the Highway 40 . Colombia and Venezuela (
edit ) The southern part of the highway begins in northwestern Colombia , from where it follows Colombia Highway 62 to Medellín .
</item 7>

<item 8>
Pan-American Highway
The first conference regarding construction of the highway occurred on October 5 , 1925 . Finally , on July 29 , 1937 , Argentina , Bolivia , Chile , Colombia ,
Costa Rica , El Salvador , Guatemala , Honduras , Mexico , Nicaragua , Panama , Peru , Canada , and the United States signed the Convention on the Pan-American
Highway , whereby they agreed to speedy construction , by all adequate means . In 1950 , Mexico became the first Latin American country to complete its portion
of the highway . Countries served ( edit ) Map of the Alaska Highway portion ( in red ) of the Pan-American Highway system . The Northern Pan-American Highway
travels through 9 countries : Canada ( unofficial ) United States ( interstate system official ) Mexico Guatemala El Salvador Honduras Nicaragua Costa Rica
Panama The Southern Pan-American Highway travels through 8 countries : Suriname Guyana Venezuela Colombia Ecuador Peru Chile Argentina Important spurs also lead
into 4 countries : Bolivia Brazil Paraguay Uruguay Hemispheric overview ( edit ) The Pan-American Highway system is physically mostly complete and extends in de
facto terms from Prudhoe Bay , Alaska , in North America to the lower reaches of South America .
</item 8>

<item 9>
Pan-American Highway
The Independent ( UK ) . Retrieved 3 August 2010 . Sources ( edit ) Plan Federal Highway System , New York Times May 15 , 1932 page XX7 Reported from the Motor
World , New York Times January 26 , 1936 page XX6 Hemisphere Road is Nearer Reality , New York Times January 7 , 1953 page 58 1997 -- 98 AAA Caribbean , Central
America and South America map Popular Mechanics , March 1943 , Longest Road In The World External links ( edit ) Media related to Pan-American Highway at
Wikimedia Commons Pan-American Highway travel guide from Wikivoyage Geographic data related to Pan-American Highway at OpenStreetMap Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Pan-American_Highway&oldid=838034809 '' Categories : Pan-American Highway Hidden categories : Articles containing
French - language text Articles containing Portuguese - language text Articles containing Spanish - language text Webarchive template wayback links CS1 Spanish
- language sources ( es ) All articles with dead external links Articles with dead external links from March 2018 Articles with permanently dead external links
Pages using div col with deprecated parameters All articles with unsourced statements Articles with
</item 9>

<item 10>
Pan-American Highway
Pan-American Highway - wikipedia Pan-American Highway Jump to : navigation , search `` Panamericana '' redirects here . For the film , see Panamericana ( film )
. Not to be confused with CANAMEX Corridor . The Pan-American Highway from Prudhoe Bay , U.S.A. to Quellón , Chile and Ushuaia , Argentina , with official and
unofficial routes shown in South America , and a few selected unofficial routes shown through the United States and Canada as they existed in the early 1960 's
. In 1966 the new U.S. Interstate highway system brought official status to most previously unofficial routes in the lower 48 states . The Pan-American Highway
is a network of roads measuring about 30,000 kilometres ( 19,000 mi ) in total length . Except for a rainforest break of approximately 160 km ( 100 mi ) ,
called the Darién Gap , the road links almost all of the Pacific coastal countries of the Americas in a connected highway system . According to Guinness World
Records , the Pan-American Highway is the world 's longest `` motorable road '' .
</item 10>
=== END ITEM id=633cad57cf12417f ===
