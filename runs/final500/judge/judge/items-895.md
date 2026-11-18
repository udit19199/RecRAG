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

=== ITEM id=316a5d85f97d0f7e ===
Gold sentences and facts:
Sentence 0: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting , a feat matched only by fellow Australian Heath Ledger 32 years later ; Finch had suffered a fatal heart attack in mid-January . Beatrice Straight set another record by becoming the actor with the shortest performance ever in a film to win an acting Oscar , with only five minutes and two seconds of screen - time in Network . Network , along with All the President 's Men , were the two biggest champs of the ceremony with four Oscars each , but Best Picture and Best Director ultimately went to Rocky .
  Fact 0.0: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting.
  Fact 0.1: Peter Finch's posthumous acting Oscar was matched only by fellow Australian Heath Ledger, 32 years later.
  Fact 0.2: Peter Finch had suffered a fatal heart attack in mid-January.
  Fact 0.3: Beatrice Straight set a record as the actor with the shortest performance ever in a film to win an acting Oscar.
  Fact 0.4: Beatrice Straight's performance in Network had only five minutes and two seconds of screen time.
  Fact 0.5: Beatrice Straight won an acting Oscar for her performance in Network.
  Fact 0.6: Network and All the President's Men were the two biggest winners of the ceremony.
  Fact 0.7: Network won four Oscars.
  Fact 0.8: All the President's Men won four Oscars.
  Fact 0.9: Best Picture and Best Director ultimately went to Rocky.

Retrieved items:
<item 1>
49th Academy Awards
Christian Szell Burt Young -- Rocky as Paulie Pennino Beatrice Straight -- Network as Louise Schumacher Jane Alexander -- All the President 's Men as Judy
Graham Hoback Jodie Foster -- Taxi Driver as Iris Steensma Lee Grant -- Voyage of the Damned as Lillian Rosen Piper Laurie -- Carrie as Margaret White Best
Screenplay Written Directly for the Screen Best Screenplay Based on Material from Another Medium Network -- Paddy Chayefsky Cousin Cousine -- Jean - Charles
Tacchella and Danièle Thompson The Front -- Walter Bernstein Rocky -- Sylvester Stallone Seven Beauties -- Lina Wertmüller All the President 's Men -- William
Goldman based on the book by Carl Bernstein and Bob Woodward Bound for Glory -- Robert Getchell based on the book by Woody Guthrie Fellini 's Casanova --
Federico Fellini and Bernardino Zapponi based on the autobiography Histoire de ma vie by Giacomo Casanova The Seven - Per - Cent Solution -- Nicholas Meyer
based on his novel Voyage of the Damned -- David Butler and Steve Shagan based on the book by Gordon Thomas and Max Morgan Witts Best Documentary Feature Best
Documentary Short Subject Harlan County , USA -- Barbara
</item 1>

<item 2>
49th Academy Awards
Kline Logan 's Run -- Ernest Laszlo Network -- Owen Roizman A Star Is Born -- Robert L. Surtees Best Film Editing Rocky -- Richard Halsey and Scott Conrad All
the President 's Men -- Robert L. Wolfe Bound for Glory -- Robert C. Jones and Pembroke J. Herring Network -- Alan Heim Two - Minute Warning -- Eve Newman and
Walter Hannemann Special Achievement Award ( edit ) Carlo Rambaldi , Glen Robinson , and Frank Van der Veer for the visual effects of King Kong L.B.
</item 2>

<item 3>
The 49th Academy Awards took place March 28, 1977, at Los Angeles’s Dorothy Chandler Pavilion, hosted by Richard Pryor, Ellen Burstyn, Jane Fonda, and Warren
Beatty. *Rocky* won Best Picture and Best Director for John G. Avildsen. *Network* and *All the President’s Men* led with four wins each. Peter Finch became the
first posthumous acting Oscar winner; Beatrice Straight won for a 5-minute performance
</item 3>

<item 4>
Summer of '42
On this point , Raucher said in May 2002 that his ten percent of the gross , in addition to royalties from book sales , `` has paid bills ever since . '' As
well as being a commercial success , Summer of ' 42 also received rave critical reviews . It went on to be nominated for over a dozen awards , including Golden
Globe Awards for `` Best Motion Picture -- Drama '' and `` Best Director '' , and five Academy Award nominations for Best Original Music Score , Best
Cinematography , Best Editing , Best Writing - Story and Best Screenplay . Ultimately , the film won two awards : the 1972 44th Academy Awards Oscar for
Original Dramatic Score , and the 1971 BAFTA Anthony Asquith Award for Film Music , both to Michel Legrand . It counted among its fans Stanley Kubrick , who had
the film play on a television in a scene in The Shining .
</item 4>

<item 5>
Gerald Ford
Buckley Ronald Reagan Harold Stassen Third party and independent candidates American Party Nominee Thomas J. Anderson American Independent Party Nominee Lester
Maddox Communist Party Nominee Gus Hall VP nominee Jarvis Tyner Libertarian Party Nominee Roger MacBride VP nominee David Bergland People 's Party Nominee
Margaret Wright VP nominee Benjamin Spock Prohibition Party Nominee Ben Bubar VP nominee Earl Dodge Socialist Workers Party Nominee Peter Camejo VP nominee
Willie Mae Reid U.S. Labor Party Nominee Lyndon LaRouche Other 1976 elections : House Senate Gubernatorial Theodore Roosevelt Award winners 1967 : Eisenhower
1968 : Saltonstall 1969 : White 1970 : Hovde 1971 : Kraft Jr . 1972 : Holland 1973 : Bradley 1974 : Owens 1975 : Ford 1976 : Hamilton 1977 : Bradley 1978 :
Zornow 1979 : Chandler 1980 : Cooley 1981 : Linkletter 1982 : Cosby 1983 : Palmer 1984 : Lawrence 1985 : Fleming 1986 : Bush 1987 : Zable 1988 : Not presented
1989 : Ebert 1990 :
</item 5>

<item 6>
49th Academy Awards
The two events conflicted again five years later in 1982 . Contents ( hide ) 1 Winners and nominees 1.1 Special Achievement Award 1.2 Irving G. Thalberg
Memorial Award 1.3 Multiple nominations and awards 2 Presenters and performers 2.1 Presenters 2.2 Performers 3 See also 4 References Winners and Nominees ( edit
) Peter Finch , Best Actor winner Faye Dunaway , Best Actress winner Jason Robards , Best Supporting Actor winner Beatrice Straight , Best Supporting Actress
winner Winners are listed first , highlighted in boldface and indicated with a double dagger ( ) . Best Picture Best Director Rocky -- Robert Chartoff and Irwin
Winkler , producers All the President 's Men -- Walter Coblenz , producer Bound for Glory -- Robert F. Blumofe and Harold Leventhal , producers Network --
Howard Gottfried , producer Taxi Driver -- Julia Phillips and Michael Phillips , producers John G. Avildsen -- Rocky Alan J.
</item 6>

<item 7>
49th Academy Awards
USA -- Barbara Kopple Hollywood on Trial -- David Helpern Off the Edge -- Michael Firth People of the Wind -- Anthony Howarth and David Koff Volcano : An
Inquiry into the Life and Death of Malcolm Lowry -- Donald Brittain and John Kramer Number Our Days -- Lynne Littman and Barbara Myerhoff American Shoeshine
Blackwood The End of the Road Universe Best Live Action Short Film Best Animated Short Film In the Region of Ice -- Peter Werner Kudzu -- Marjorie Anne Short
The Morning Spider -- Julian Chagrin and Claude Chagrin Nightlife -- Claire Wilbur and Robin Lehman Number One -- Dyan Cannon and Vince Cannon Leisure --
Suzanne Baker Dedalo -- Manfredo Manfredi The Street -- Caroline Leaf and Guy Glover Best Original Score Best Original Song Score or Adaptation Score The Omen
-- Jerry Goldsmith Obsession -- Bernard Herrmann ( posthumous nomination ) The Outlaw Josey Wales -- Jerry Fielding Taxi Driver -- Bernard Herrmann ( posthumous
nomination ) Voyage of the Damned -- Lalo Schifrin Bound for Glory -- Leonard Rosenman Bugsy Malone -- Paul Williams A Star Is Born -- Roger Kellaway Best
Original Song Best Sound ``
</item 7>

<item 8>
49th Academy Awards
producers John G. Avildsen -- Rocky Alan J. Pakula -- All the President 's Men Ingmar Bergman -- Face to Face Sidney Lumet -- Network Lina Wertmüller -- Seven
Beauties Best Actor Best Actress Peter Finch -- Network as Howard Beale Robert De Niro -- Taxi Driver as Travis Bickle Giancarlo Giannini -- Seven Beauties as
Pasqualino William Holden -- Network as Max Schumacher Sylvester Stallone -- Rocky as Rocky Balboa Faye Dunaway -- Network as Diana Christensen Marie -
Christine Barrault -- Cousin Cousine as Marthe Talia Shire -- Rocky as Adriana Pennino Sissy Spacek -- Carrie as Carrie White Liv Ullmann -- Face to Face as
Jenny Isaksson Best Supporting Actor Best Supporting Actress Jason Robards -- All the President 's Men as Ben Bradlee Ned Beatty -- Network as Arthur Jensen
Burgess Meredith -- Rocky as `` Mickey '' Goldmill Laurence Olivier -- Marathon Man as Dr.
</item 8>

<item 9>
The Deer Hunter
In addition , the film was nominated in four other categories : Best Actor in a Leading Role -- Robert De Niro ( lost to Jon Voight for Coming Home ) Best
Actress in a Supporting Role -- Meryl Streep ( lost to Maggie Smith for California Suite ) Best Cinematography -- Vilmos Zsigmond ( lost to Néstor Almendros for
Days of Heaven ) Best Writing , Screenplay Written Directly for the Screen -- Michael Cimino , Deric Washburn , Louis Garfinkle and Quinn Redeker ( lost to
Robert C. Jones , Waldo Salt and Nancy Dowd for Coming Home ) Golden globes ( edit ) Cimino won the film 's only Golden Globe Award for Best Director . Other
nominations the film included Best Motion Picture -- Drama , De Niro for Best Motion Picture Actor -- Drama , Walken for Best Motion Picture Actor in a
Supporting Role , Streep for Best Motion Picture Actress in a Supporting Role , and Washburn for Best Screenplay -- Motion Picture . Complete list of awards (
edit ) Award Recipients and nominees Result 51st Academy Awards Academy Award for Best Picture Barry Spikings , Michael Deeley ,
</item 9>

<item 10>
*Summer of ’42* was both a commercial and critical success. Screenwriter Herman Raucher said his 10% share of gross, plus book royalties, continued to pay his
bills. The film received more than a dozen award nominations, including two Golden Globe and five Oscar nominations, and won the 1972 Oscar and 1971 BAFTA for
Michel Legrand’s music. Stanley Kubrick was a fan and featured it on a television in *The Shining*.
</item 10>
=== END ITEM id=316a5d85f97d0f7e ===

=== ITEM id=b5b1157936726767 ===
Gold sentences and facts:
Sentence 0: ABO and Rh blood type distribution by country ( population averages ) Country Population O+ A+ B+ AB+ O − A − B − AB − Argentina 44,270,440 45.4 % 34.26 % 8.59 % 2.64 % 8.4 % 0.44 % 0.21 % 0.06 % Armenia 2,931,568 29.0 % 46.3 % 12.0 % 5.6 % 2.0 % 3.7 % 1.0 % 0.4 % Australia 24,642,693 40.0 % 31.0 % 8.0 % 2.0 % 9.0 % 7.0 % 2.0 % 1.0 % Austria 8,592,470 30.0 % 33.0 % 12.0 % 6.0 % 7.0 % 8.0 % 3.0 % 1.0 % Bahrain 1,418,695 48.48 % 19.35 % 22.61 % 3.67 % 3.27 % 1.33 % 1.04 % 0.25 % Bangladesh 164,833,667 31.18 % 21.44 % 34.58 % 8.85 % 1.39 % 0.96 % 0.96 % 0.64 % Belgium 11,444,053 38.0 % 34.0 % 8.6 % 4.1 % 7.0 % 6.0 % 1.5 % 0.8 % Bolivia 11,053,376 51.53 % 29.45 % 10.11 % 1.15 % 4.39 % 2.73 % 0.54 % 0.1 % Bosnia and Herzegovina 3,792,730 31.0 % 36.0 % 12.0 % 6.0 % 5.0 % 7.0 % 2.0 % 1.0 % Brazil 211,248,418 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Bulgaria 7,045,097 28.0 % 37.0 % 13.0 % 7.0 % 5.0 % 7.0 % 2.0 % 1.0 % Cambodia 16,077,172 46.7 % 27.2 % 18.5 % 4.9 % 1.3 % 0.8 % 0.5 % 0.1 % Cameroon 24,515,533 42.8 % 38.8 % 12.0 % 3.3 % 1.4 % 1.2 % 0.4 % 0.1 % Canada 36,627,140 39.0 % 36.0 % 7.6 % 2.5 % 7.0 % 6.0 % 1.4 % 0.5 % Chile 18,314,060 85.5 % 8.7 % 3.35 % 1.0 % 1.2 % 0.1 % 0.05 % 0.1 % China 1,388,251,023 47.7 % 27.8 % 18.9 % 5.0 % 0.28 % 0.19 % 0.1 % 0.03 % Colombia 49,069,267 61.3 % 26.11 % 2.28 % 1.47 % 5.13 % 2.7 % 0.7 % 0.31 % Côte d'Ivoire 23,869,656 46.5 % 22.5 % 22.5 % 4.3 % 2.0 % 1.0 % 1.0 % 0.2 % Croatia 4,207,355 29.0 % 36.0 % 15.0 % 5.0 % 5.0 % 6.0 % 3.0 % 1.0 % Cuba 11,486,750 45.8 % 33.5 % 10.2 % 2.9 % 3.6 % 2.8 % 1.0 % 0.2 % Cyprus 1,189,395 35.22 % 40.35 % 11.11 % 4.72 % 3.85 % 3.48 % 0.87 % 0.40 % Czech Republic 10,555,152 27.0 % 36.0 % 15.0 % 7.0 % 5.0 % 6.0 % 3.0 % 1.0 % Denmark 5,711,902 35.0 % 37.0 % 8.0 % 4.0 % 6.0 % 7.0 % 2.0 % 1.0 % Dominican Republic 10,766,932 46.2 % 26.4 % 16.9 % 3.1 % 3.7 % 2.1 % 1.4 % 0.2 % Ecuador 16,665,756 75.0 % 14.0 % 7.1 % 0.5 % 2.38 % 0.7 % 0.3 % 0.02 % Egypt 95,220,838 52.0 % 24.0 % 12.4 % 3.8 % 5.0 % 2.0 % 0.6 % 0.2 % El Salvador 6,171,483 62.0 % 23.0 % 11.0 % 1.0 % 1.0 % 1.0 % 0.7 % 0.3 % Estonia 1,305,745 29.5 % 30.8 % 20.7 % 6.3 % 4.3 % 4.5 % 3.0 % 0.9 % Ethiopia 104,352,727 39.0 % 28.0 % 21.0 % 5.0 % 3.0 % 2.0 % 1.0 % 1.0 % Finland 5,541,328 27.0 % 38.0 % 15.0 % 7.0 % 4.0 % 6.0 % 2.0 % 1.0 % France 64,939,560 36.0 % 37.0 % 9.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Germany 80,635,980 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Greece 10,892,849 37.4 % 32.9 % 11.0 % 3.7 % 7.0 % 5.0 % 2.0 % 1.0 % Guinea 13,291,741 46.88 % 21.64 % 22.86 % 4.52 % 2.0 % 0.9 % 1.0 % 0.2 % Honduras 9,348,898 57.5 % 27.0 % 7.8 % 2.5 % 2.7 % 1.7 % 0.6 % 0.2 % Hong Kong 7,402,115 41.5 % 26.13 % 25.34 % 6.35 % 0.32 % 0.17 % 0.14 % 0.05 % Hungary 9,787,801 27.0 % 33.0 % 16.0 % 8.0 % 5.0 % 7.0 % 3.0 % 1.0 % Iceland 334,311 47.6 % 26.4 % 9.3 % 1.6 % 8.4 % 4.6 % 1.7 % 0.4 % India 1,342,561,902 27.85 % 20.8 % 38.14 % 8.93 % 1.43 % 0.57 % 1.79 % 0.49 % Indonesia 263,519,317 36.82 % 25.87 % 28.85 % 7.96 % 0.18 % 0.13 % 0.15 % 0.04 % Iran 80,982,137 33.5 % 27.0 % 22.2 % 7.0 % 4.0 % 3.0 % 2.5 % 0.8 % Iraq 38,657,787 32.1 % 25.0 % 25.6 % 7.4 % 3.6 % 2.7 % 2.7 % 0.9 % Ireland 4,749,263 47.0 % 26.0 % 9.0 % 2.0 % 8.0 % 5.0 % 2.0 % 1.0 % Israel 8,323,659 32.0 % 34.0 % 17.0 % 7.0 % 3.0 % 4.0 % 2.0 % 1.0 % Italy 59,797,969 39.0 % 36.0 % 7.5 % 2.5 % 7.0 % 6.0 % 1.5 % 0.5 % Jamaica 2,813,316 47.0 % 23.0 % 20.0 % 3.0 % 3.5 % 2.0 % 1.0 % 0.5 % Japan 126,044,340 29.9 % 39.8 % 19.9 % 9.9 % 0.15 % 0.2 % 0.1 % 0.05 % Kenya 48,470,780 45.6 % 25.2 % 21.28 % 4.2 % 1.8 % 1.0 % 0.9 % 0.02 % Latvia 1,944,122 30.6 % 31.0 % 17.0 % 6.0 % 5.4 % 6.0 % 3.0 % 1.0 % Lebanon 6,039,438 38.4 % 32.3 % 9.5 % 3.2 % 7.7 % 6.5 % 1.7 % 0.7 % Libya 6,408,990 42.64 % 28.86 % 11.19 % 4.5 % 6.26 % 4.24 % 1.64 % 0.67 % Lithuania 2,827,573 36.0 % 33.0 % 11.0 % 4.0 % 7.0 % 6.0 % 2.0 % 1.0 % Luxemburg 587,297 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Macao 606,413 41.5 % 26.1 % 25.4 % 6.3 % 0.33 % 0.17 % 0.15 % 0.05 % Macedonia 2,083,690 30.0 % 34.0 % 15.0 % 6.0 % 5.0 % 6.0 % 3.0 % 1.0 % Malaysia 31,165,480 34.32 % 30.35 % 27.37 % 7.46 % 0.17 % 0.15 % 0.14 % 0.04 % Mauritania 4,266,765 46.3 % 26.68 % 17.47 % 3.85 % 2.8 % 1.6 % 1.1 % 0.2 % Mauritius 1,267,303 38.3 % 26.0 % 25.0 % 6.7 % 1.7 % 1.0 % 1.0 % 0.3 % Mexico 130,227,836 55.79 % 29.9 % 8.0 % 1.63 % 2.7 % 1.5 % 0.4 % 0.08 % Mongolia 3,052,044 55.4 % 27.7 % 10.9 % 4.96 % 0.6 % 0.3 % 0.1 % 0.04 % Myanmar 53,507,932 35.7 % 23.8 % 32.7 % 6.95 % 0.3 % 0.2 % 0.3 % 0.05 % Nepal 29,188,100 35.2 % 28.3 % 27.1 % 8.6 % 0.3 % 0.2 % 0.2 % 0.1 % Netherlands 17,033,012 39.5 % 35.0 % 6.7 % 2.5 % 7.5 % 7.0 % 1.3 % 0.5 % New Zealand 4,604,996 38.0 % 32.0 % 9.0 % 3.0 % 9.0 % 6.0 % 2.0 % 1.0 % Nigeria 191,851,411 51.3 % 22.4 % 20.7 % 2.6 % 1.6 % 0.7 % 0.6 % 0.1 % Norway 5,330,986 33.0 % 41.5 % 6.8 % 3.4 % 6.0 % 7.5 % 1.2 % 0.6 % North Korea 25,432,033 27.15 % 31.08 % 30.15 % 11.32 % 0.08 % 0.1 % 0.1 % 0.03 % Pakistan 180,440,005 24.63 % 20.6 % 34.40 % 9.52 % 4.17 % 2.66 % 3.57 % 0.45 % Papua New Guinea 7,952,511 48.0 % 37.2 % 9.6 % 2.8 % 1.0 % 0.8 % 0.4 % 0.2 % Peru 32,167,717 70.0 % 18.4 % 7.8 % 1.6 % 1.4 % 0.5 % 0.28 % 0.02 % Philippines 103,801,747 36.9 % 28.9 % 26.9 % 6.97 % 0.1 % 0.1 % 0.1 % 0.03 % Poland 38,563,480 31.0 % 32.0 % 15.0 % 7.0 % 6.0 % 6.0 % 2.0 % 1.0 % Portugal 10,264,672 36.3 % 40.0 % 6.6 % 2.9 % 6.0 % 6.6 % 1.1 % 0.5 % Republic of Moldova 4,043,722 28.5 % 31.8 % 17.6 % 7.0 % 5.0 % 6.0 % 3.0 % 1.1 % Romania 19,237,087 28.0 % 37.0 % 14.0 % 7.0 % 5.0 % 6.0 % 2.0 % 1.0 % Russia 143,374,801 28.0 % 30.0 % 20.0 % 7.0 % 4.9 % 5.8 % 3.2 % 1.1 % Saudi Arabia 32,744,532 47.8 % 23.9 % 17.0 % 4.0 % 4.0 % 2.0 % 1.0 % 0.3 % Serbia 8,776,827 31.92 % 35.28 % 12.6 % 4.2 % 6.08 % 6.72 % 2.4 % 0.8 % Singapore 5,784,819 43.6 % 23.9 % 24.4 % 6.0 % 0.9 % 0.7 % 0.4 % 0.1 % Slovakia 5,432,265 27.2 % 35.7 % 15.3 % 6.8 % 4.8 % 6.3 % 2.7 % 1.2 % Slovenia 2,071,258 31.0 % 33.0 % 12.0 % 6.0 % 7.0 % 7.0 % 3.0 % 1.0 % South Africa 55,437,815 39.0 % 32.0 % 12.0 % 3.0 % 6.0 % 5.0 % 2.0 % 1.0 % South Korea 50,748,307 27.9 % 33.87 % 26.92 % 10.98 % 0.1 % 0.13 % 0.08 % 0.02 % Spain 46,070,163 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Sudan 40,827,975 48.0 % 27.7 % 15.2 % 2.8 % 3.5 % 1.8 % 0.8 % 0.2 % Sweden 9,920,843 32.0 % 37.0 % 10.0 % 5.0 % 6.0 % 7.0 % 2.0 % 1.0 % Switzerland 8,454,321 35.0 % 40.0 % 7.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Syria 18,920,698 43.0 % 30.0 % 14.0 % 3.7 % 5.0 % 3.0 % 1.0 % 0.3 % Taiwan 23,234,936 43.9 % 25.9 % 23.9 % 6.0 % 0.17 % 0.1 % 0.01 % 0.02 % Thailand 68,298,027 40.8 % 16.9 % 36.8 % 4.97 % 0.2 % 0.1 % 0.2 % 0.03 % Turkey 80,420,065 29.8 % 37.8 % 14.2 % 7.2 % 3.9 % 4.7 % 1.6 % 0.8 % Uganda 43,276,492 43.7 % 39.0 % 10.7 % 3.9 % 1.3 % 1.0 % 0.3 % 0.1 % Ukraine 44,404,078 32.0 % 34.0 % 15.0 % 5.0 % 5.0 % 6.0 % 2.0 % 1.0 % United Arab Emirates 3,398,017 44.1 % 21.9 % 20.9 % 4.3 % 4.3 % 2.1 % 2.0 % 0.4 % United Kingdom 65,512,375 37.0 % 35.0 % 8.0 % 3.0 % 7.0 % 7.0 % 2.0 % 1.0 % United States 326,481,533 37.4 % 35.7 % 8.5 % 3.4 % 6.6 % 6.3 % 1.5 % 0.6 % Venezuela 32,011,082 58.5 % 28.3 % 5.3 % 1.8 % 3.5 % 1.7 % 0.7 % 0.2 % Vietnam 95,414,640 41.7 % 21.9 % 30.8 % 4.98 % 0.3 % 0.1 % 0.2 % 0.02 % Zimbabwe 16,642,880 63.3 % 18.5 % 14.6 % 0.99 % 1.7 % 0.5 % 0.4 % 0.01 % World 6,635,281,257 38.67 % 27.42 % 22.02 % 5.88 % 2.55 % 1.99 % 1.11 % 0.36 %
  Fact 0.0: The table gives ABO and Rh blood type distribution by country as population averages.
  Fact 0.1: Each row lists a country, its population, and the percentages of O+, A+, B+, AB+, O−, A−, B−, and AB− blood types.
  Fact 0.2: Argentina has a population of 44,270,440, with blood type percentages O+ 45.4, A+ 34.26, B+ 8.59, AB+ 2.64, O− 8.4, A− 0.44, B− 0.21, AB− 0.06.
  Fact 0.3: Armenia has a population of 2,931,568, with blood type percentages O+ 29.0, A+ 46.3, B+ 12.0, AB+ 5.6, O− 2.0, A− 3.7, B− 1.0, AB− 0.4.
  Fact 0.4: Australia has a population of 24,642,693, with blood type percentages O+ 40.0, A+ 31.0, B+ 8.0, AB+ 2.0, O− 9.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.5: Austria has a population of 8,592,470, with blood type percentages O+ 30.0, A+ 33.0, B+ 12.0, AB+ 6.0, O− 7.0, A− 8.0, B− 3.0, AB− 1.0.
  Fact 0.6: Bahrain has a population of 1,418,695, with blood type percentages O+ 48.48, A+ 19.35, B+ 22.61, AB+ 3.67, O− 3.27, A− 1.33, B− 1.04, AB− 0.25.
  Fact 0.7: Bangladesh has a population of 164,833,667, with blood type percentages O+ 31.18, A+ 21.44, B+ 34.58, AB+ 8.85, O− 1.39, A− 0.96, B− 0.96, AB− 0.64.
  Fact 0.8: Belgium has a population of 11,444,053, with blood type percentages O+ 38.0, A+ 34.0, B+ 8.6, AB+ 4.1, O− 7.0, A− 6.0, B− 1.5, AB− 0.8.
  Fact 0.9: Bolivia has a population of 11,053,376, with blood type percentages O+ 51.53, A+ 29.45, B+ 10.11, AB+ 1.15, O− 4.39, A− 2.73, B− 0.54, AB− 0.1.
  Fact 0.10: Bosnia and Herzegovina has a population of 3,792,730, with blood type percentages O+ 31.0, A+ 36.0, B+ 12.0, AB+ 6.0, O− 5.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.11: Brazil has a population of 211,248,418, with blood type percentages O+ 36.0, A+ 34.0, B+ 8.0, AB+ 2.5, O− 9.0, A− 8.0, B− 2.0, AB− 0.5.
  Fact 0.12: Bulgaria has a population of 7,045,097, with blood type percentages O+ 28.0, A+ 37.0, B+ 13.0, AB+ 7.0, O− 5.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.13: Cambodia has a population of 16,077,172, with blood type percentages O+ 46.7, A+ 27.2, B+ 18.5, AB+ 4.9, O− 1.3, A− 0.8, B− 0.5, AB− 0.1.
  Fact 0.14: Cameroon has a population of 24,515,533, with blood type percentages O+ 42.8, A+ 38.8, B+ 12.0, AB+ 3.3, O− 1.4, A− 1.2, B− 0.4, AB− 0.1.
  Fact 0.15: Canada has a population of 36,627,140, with blood type percentages O+ 39.0, A+ 36.0, B+ 7.6, AB+ 2.5, O− 7.0, A− 6.0, B− 1.4, AB− 0.5.
  Fact 0.16: Chile has a population of 18,314,060, with blood type percentages O+ 85.5, A+ 8.7, B+ 3.35, AB+ 1.0, O− 1.2, A− 0.1, B− 0.05, AB− 0.1.
  Fact 0.17: China has a population of 1,388,251,023, with blood type percentages O+ 47.7, A+ 27.8, B+ 18.9, AB+ 5.0, O− 0.28, A− 0.19, B− 0.1, AB− 0.03.
  Fact 0.18: Colombia has a population of 49,069,267, with blood type percentages O+ 61.3, A+ 26.11, B+ 2.28, AB+ 1.47, O− 5.13, A− 2.7, B− 0.7, AB− 0.31.
  Fact 0.19: Côte d'Ivoire has a population of 23,869,656, with blood type percentages O+ 46.5, A+ 22.5, B+ 22.5, AB+ 4.3, O− 2.0, A− 1.0, B− 1.0, AB− 0.2.
  Fact 0.20: Croatia has a population of 4,207,355, with blood type percentages O+ 29.0, A+ 36.0, B+ 15.0, AB+ 5.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.21: Cuba has a population of 11,486,750, with blood type percentages O+ 45.8, A+ 33.5, B+ 10.2, AB+ 2.9, O− 3.6, A− 2.8, B− 1.0, AB− 0.2.
  Fact 0.22: Cyprus has a population of 1,189,395, with blood type percentages O+ 35.22, A+ 40.35, B+ 11.11, AB+ 4.72, O− 3.85, A− 3.48, B− 0.87, AB− 0.40.
  Fact 0.23: Czech Republic has a population of 10,555,152, with blood type percentages O+ 27.0, A+ 36.0, B+ 15.0, AB+ 7.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.24: Denmark has a population of 5,711,902, with blood type percentages O+ 35.0, A+ 37.0, B+ 8.0, AB+ 4.0, O− 6.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.25: Dominican Republic has a population of 10,766,932, with blood type percentages O+ 46.2, A+ 26.4, B+ 16.9, AB+ 3.1, O− 3.7, A− 2.1, B− 1.4, AB− 0.2.
  Fact 0.26: Ecuador has a population of 16,665,756, with blood type percentages O+ 75.0, A+ 14.0, B+ 7.1, AB+ 0.5, O− 2.38, A− 0.7, B− 0.3, AB− 0.02.
  Fact 0.27: Egypt has a population of 95,220,838, with blood type percentages O+ 52.0, A+ 24.0, B+ 12.4, AB+ 3.8, O− 5.0, A− 2.0, B− 0.6, AB− 0.2.
  Fact 0.28: El Salvador has a population of 6,171,483, with blood type percentages O+ 62.0, A+ 23.0, B+ 11.0, AB+ 1.0, O− 1.0, A− 1.0, B− 0.7, AB− 0.3.
  Fact 0.29: Estonia has a population of 1,305,745, with blood type percentages O+ 29.5, A+ 30.8, B+ 20.7, AB+ 6.3, O− 4.3, A− 4.5, B− 3.0, AB− 0.9.
  Fact 0.30: Ethiopia has a population of 104,352,727, with blood type percentages O+ 39.0, A+ 28.0, B+ 21.0, AB+ 5.0, O− 3.0, A− 2.0, B− 1.0, AB− 1.0.
  Fact 0.31: Finland has a population of 5,541,328, with blood type percentages O+ 27.0, A+ 38.0, B+ 15.0, AB+ 7.0, O− 4.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.32: France has a population of 64,939,560, with blood type percentages O+ 36.0, A+ 37.0, B+ 9.0, AB+ 3.0, O− 6.0, A− 7.0, B− 1.0, AB− 1.0.
  Fact 0.33: Germany has a population of 80,635,980, with blood type percentages O+ 35.0, A+ 37.0, B+ 9.0, AB+ 4.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.34: Greece has a population of 10,892,849, with blood type percentages O+ 37.4, A+ 32.9, B+ 11.0, AB+ 3.7, O− 7.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.35: Guinea has a population of 13,291,741, with blood type percentages O+ 46.88, A+ 21.64, B+ 22.86, AB+ 4.52, O− 2.0, A− 0.9, B− 1.0, AB− 0.2.
  Fact 0.36: Honduras has a population of 9,348,898, with blood type percentages O+ 57.5, A+ 27.0, B+ 7.8, AB+ 2.5, O− 2.7, A− 1.7, B− 0.6, AB− 0.2.
  Fact 0.37: Hong Kong has a population of 7,402,115, with blood type percentages O+ 41.5, A+ 26.13, B+ 25.34, AB+ 6.35, O− 0.32, A− 0.17, B− 0.14, AB− 0.05.
  Fact 0.38: Hungary has a population of 9,787,801, with blood type percentages O+ 27.0, A+ 33.0, B+ 16.0, AB+ 8.0, O− 5.0, A− 7.0, B− 3.0, AB− 1.0.
  Fact 0.39: Iceland has a population of 334,311, with blood type percentages O+ 47.6, A+ 26.4, B+ 9.3, AB+ 1.6, O− 8.4, A− 4.6, B− 1.7, AB− 0.4.
  Fact 0.40: India has a population of 1,342,561,902, with blood type percentages O+ 27.85, A+ 20.8, B+ 38.14, AB+ 8.93, O− 1.43, A− 0.57, B− 1.79, AB− 0.49.
  Fact 0.41: Indonesia has a population of 263,519,317, with blood type percentages O+ 36.82, A+ 25.87, B+ 28.85, AB+ 7.96, O− 0.18, A− 0.13, B− 0.15, AB− 0.04.
  Fact 0.42: Iran has a population of 80,982,137, with blood type percentages O+ 33.5, A+ 27.0, B+ 22.2, AB+ 7.0, O− 4.0, A− 3.0, B− 2.5, AB− 0.8.
  Fact 0.43: Iraq has a population of 38,657,787, with blood type percentages O+ 32.1, A+ 25.0, B+ 25.6, AB+ 7.4, O− 3.6, A− 2.7, B− 2.7, AB− 0.9.
  Fact 0.44: Ireland has a population of 4,749,263, with blood type percentages O+ 47.0, A+ 26.0, B+ 9.0, AB+ 2.0, O− 8.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.45: Israel has a population of 8,323,659, with blood type percentages O+ 32.0, A+ 34.0, B+ 17.0, AB+ 7.0, O− 3.0, A− 4.0, B− 2.0, AB− 1.0.
  Fact 0.46: Italy has a population of 59,797,969, with blood type percentages O+ 39.0, A+ 36.0, B+ 7.5, AB+ 2.5, O− 7.0, A− 6.0, B− 1.5, AB− 0.5.
  Fact 0.47: Jamaica has a population of 2,813,316, with blood type percentages O+ 47.0, A+ 23.0, B+ 20.0, AB+ 3.0, O− 3.5, A− 2.0, B− 1.0, AB− 0.5.
  Fact 0.48: Japan has a population of 126,044,340, with blood type percentages O+ 29.9, A+ 39.8, B+ 19.9, AB+ 9.9, O− 0.15, A− 0.2, B− 0.1, AB− 0.05.
  Fact 0.49: Kenya has a population of 48,470,780, with blood type percentages O+ 45.6, A+ 25.2, B+ 21.28, AB+ 4.2, O− 1.8, A− 1.0, B− 0.9, AB− 0.02.
  Fact 0.50: Latvia has a population of 1,944,122, with blood type percentages O+ 30.6, A+ 31.0, B+ 17.0, AB+ 6.0, O− 5.4, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.51: Lebanon has a population of 6,039,438, with blood type percentages O+ 38.4, A+ 32.3, B+ 9.5, AB+ 3.2, O− 7.7, A− 6.5, B− 1.7, AB− 0.7.
  Fact 0.52: Libya has a population of 6,408,990, with blood type percentages O+ 42.64, A+ 28.86, B+ 11.19, AB+ 4.5, O− 6.26, A− 4.24, B− 1.64, AB− 0.67.
  Fact 0.53: Lithuania has a population of 2,827,573, with blood type percentages O+ 36.0, A+ 33.0, B+ 11.0, AB+ 4.0, O− 7.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.54: Luxemburg has a population of 587,297, with blood type percentages O+ 35.0, A+ 37.0, B+ 9.0, AB+ 4.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.55: Macao has a population of 606,413, with blood type percentages O+ 41.5, A+ 26.1, B+ 25.4, AB+ 6.3, O− 0.33, A− 0.17, B− 0.15, AB− 0.05.
  Fact 0.56: Macedonia has a population of 2,083,690, with blood type percentages O+ 30.0, A+ 34.0, B+ 15.0, AB+ 6.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.57: Malaysia has a population of 31,165,480, with blood type percentages O+ 34.32, A+ 30.35, B+ 27.37, AB+ 7.46, O− 0.17, A− 0.15, B− 0.14, AB− 0.04.
  Fact 0.58: Mauritania has a population of 4,266,765, with blood type percentages O+ 46.3, A+ 26.68, B+ 17.47, AB+ 3.85, O− 2.8, A− 1.6, B− 1.1, AB− 0.2.
  Fact 0.59: Mauritius has a population of 1,267,303, with blood type percentages O+ 38.3, A+ 26.0, B+ 25.0, AB+ 6.7, O− 1.7, A− 1.0, B− 1.0, AB− 0.3.
  Fact 0.60: Mexico has a population of 130,227,836, with blood type percentages O+ 55.79, A+ 29.9, B+ 8.0, AB+ 1.63, O− 2.7, A− 1.5, B− 0.4, AB− 0.08.
  Fact 0.61: Mongolia has a population of 3,052,044, with blood type percentages O+ 55.4, A+ 27.7, B+ 10.9, AB+ 4.96, O− 0.6, A− 0.3, B− 0.1, AB− 0.04.
  Fact 0.62: Myanmar has a population of 53,507,932, with blood type percentages O+ 35.7, A+ 23.8, B+ 32.7, AB+ 6.95, O− 0.3, A− 0.2, B− 0.3, AB− 0.05.
  Fact 0.63: Nepal has a population of 29,188,100, with blood type percentages O+ 35.2, A+ 28.3, B+ 27.1, AB+ 8.6, O− 0.3, A− 0.2, B− 0.2, AB− 0.1.
  Fact 0.64: Netherlands has a population of 17,033,012, with blood type percentages O+ 39.5, A+ 35.0, B+ 6.7, AB+ 2.5, O− 7.5, A− 7.0, B− 1.3, AB− 0.5.
  Fact 0.65: New Zealand has a population of 4,604,996, with blood type percentages O+ 38.0, A+ 32.0, B+ 9.0, AB+ 3.0, O− 9.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.66: Nigeria has a population of 191,851,411, with blood type percentages O+ 51.3, A+ 22.4, B+ 20.7, AB+ 2.6, O− 1.6, A− 0.7, B− 0.6, AB− 0.1.
  Fact 0.67: Norway has a population of 5,330,986, with blood type percentages O+ 33.0, A+ 41.5, B+ 6.8, AB+ 3.4, O− 6.0, A− 7.5, B− 1.2, AB− 0.6.
  Fact 0.68: North Korea has a population of 25,432,033, with blood type percentages O+ 27.15, A+ 31.08, B+ 30.15, AB+ 11.32, O− 0.08, A− 0.1, B− 0.1, AB− 0.03.
  Fact 0.69: Pakistan has a population of 180,440,005, with blood type percentages O+ 24.63, A+ 20.6, B+ 34.40, AB+ 9.52, O− 4.17, A− 2.66, B− 3.57, AB− 0.45.
  Fact 0.70: Papua New Guinea has a population of 7,952,511, with blood type percentages O+ 48.0, A+ 37.2, B+ 9.6, AB+ 2.8, O− 1.0, A− 0.8, B− 0.4, AB− 0.2.
  Fact 0.71: Peru has a population of 32,167,717, with blood type percentages O+ 70.0, A+ 18.4, B+ 7.8, AB+ 1.6, O− 1.4, A− 0.5, B− 0.28, AB− 0.02.
  Fact 0.72: Philippines has a population of 103,801,747, with blood type percentages O+ 36.9, A+ 28.9, B+ 26.9, AB+ 6.97, O− 0.1, A− 0.1, B− 0.1, AB− 0.03.
  Fact 0.73: Poland has a population of 38,563,480, with blood type percentages O+ 31.0, A+ 32.0, B+ 15.0, AB+ 7.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.74: Portugal has a population of 10,264,672, with blood type percentages O+ 36.3, A+ 40.0, B+ 6.6, AB+ 2.9, O− 6.0, A− 6.6, B− 1.1, AB− 0.5.
  Fact 0.75: Republic of Moldova has a population of 4,043,722, with blood type percentages O+ 28.5, A+ 31.8, B+ 17.6, AB+ 7.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.1.
  Fact 0.76: Romania has a population of 19,237,087, with blood type percentages O+ 28.0, A+ 37.0, B+ 14.0, AB+ 7.0, O− 5.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.77: Russia has a population of 143,374,801, with blood type percentages O+ 28.0, A+ 30.0, B+ 20.0, AB+ 7.0, O− 4.9, A− 5.8, B− 3.2, AB− 1.1.
  Fact 0.78: Saudi Arabia has a population of 32,744,532, with blood type percentages O+ 47.8, A+ 23.9, B+ 17.0, AB+ 4.0, O− 4.0, A− 2.0, B− 1.0, AB− 0.3.
  Fact 0.79: Serbia has a population of 8,776,827, with blood type percentages O+ 31.92, A+ 35.28, B+ 12.6, AB+ 4.2, O− 6.08, A− 6.72, B− 2.4, AB− 0.8.
  Fact 0.80: Singapore has a population of 5,784,819, with blood type percentages O+ 43.6, A+ 23.9, B+ 24.4, AB+ 6.0, O− 0.9, A− 0.7, B− 0.4, AB− 0.1.
  Fact 0.81: Slovakia has a population of 5,432,265, with blood type percentages O+ 27.2, A+ 35.7, B+ 15.3, AB+ 6.8, O− 4.8, A− 6.3, B− 2.7, AB− 1.2.
  Fact 0.82: Slovenia has a population of 2,071,258, with blood type percentages O+ 31.0, A+ 33.0, B+ 12.0, AB+ 6.0, O− 7.0, A− 7.0, B− 3.0, AB− 1.0.
  Fact 0.83: South Africa has a population of 55,437,815, with blood type percentages O+ 39.0, A+ 32.0, B+ 12.0, AB+ 3.0, O− 6.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.84: South Korea has a population of 50,748,307, with blood type percentages O+ 27.9, A+ 33.87, B+ 26.92, AB+ 10.98, O− 0.1, A− 0.13, B− 0.08, AB− 0.02.
  Fact 0.85: Spain has a population of 46,070,163, with blood type percentages O+ 36.0, A+ 34.0, B+ 8.0, AB+ 2.5, O− 9.0, A− 8.0, B− 2.0, AB− 0.5.
  Fact 0.86: Sudan has a population of 40,827,975, with blood type percentages O+ 48.0, A+ 27.7, B+ 15.2, AB+ 2.8, O− 3.5, A− 1.8, B− 0.8, AB− 0.2.
  Fact 0.87: Sweden has a population of 9,920,843, with blood type percentages O+ 32.0, A+ 37.0, B+ 10.0, AB+ 5.0, O− 6.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.88: Switzerland has a population of 8,454,321, with blood type percentages O+ 35.0, A+ 40.0, B+ 7.0, AB+ 3.0, O− 6.0, A− 7.0, B− 1.0, AB− 1.0.
  Fact 0.89: Syria has a population of 18,920,698, with blood type percentages O+ 43.0, A+ 30.0, B+ 14.0, AB+ 3.7, O− 5.0, A− 3.0, B− 1.0, AB− 0.3.
  Fact 0.90: Taiwan has a population of 23,234,936, with blood type percentages O+ 43.9, A+ 25.9, B+ 23.9, AB+ 6.0, O− 0.17, A− 0.1, B− 0.01, AB− 0.02.
  Fact 0.91: Thailand has a population of 68,298,027, with blood type percentages O+ 40.8, A+ 16.9, B+ 36.8, AB+ 4.97, O− 0.2, A− 0.1, B− 0.2, AB− 0.03.
  Fact 0.92: Turkey has a population of 80,420,065, with blood type percentages O+ 29.8, A+ 37.8, B+ 14.2, AB+ 7.2, O− 3.9, A− 4.7, B− 1.6, AB− 0.8.
  Fact 0.93: Uganda has a population of 43,276,492, with blood type percentages O+ 43.7, A+ 39.0, B+ 10.7, AB+ 3.9, O− 1.3, A− 1.0, B− 0.3, AB− 0.1.
  Fact 0.94: Ukraine has a population of 44,404,078, with blood type percentages O+ 32.0, A+ 34.0, B+ 15.0, AB+ 5.0, O− 5.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.95: United Arab Emirates has a population of 3,398,017, with blood type percentages O+ 44.1, A+ 21.9, B+ 20.9, AB+ 4.3, O− 4.3, A− 2.1, B− 2.0, AB− 0.4.
  Fact 0.96: United Kingdom has a population of 65,512,375, with blood type percentages O+ 37.0, A+ 35.0, B+ 8.0, AB+ 3.0, O− 7.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.97: United States has a population of 326,481,533, with blood type percentages O+ 37.4, A+ 35.7, B+ 8.5, AB+ 3.4, O− 6.6, A− 6.3, B− 1.5, AB− 0.6.
  Fact 0.98: Venezuela has a population of 32,011,082, with blood type percentages O+ 58.5, A+ 28.3, B+ 5.3, AB+ 1.8, O− 3.5, A− 1.7, B− 0.7, AB− 0.2.
  Fact 0.99: Vietnam has a population of 95,414,640, with blood type percentages O+ 41.7, A+ 21.9, B+ 30.8, AB+ 4.98, O− 0.3, A− 0.1, B− 0.2, AB− 0.02.
  Fact 0.100: Zimbabwe has a population of 16,642,880, with blood type percentages O+ 63.3, A+ 18.5, B+ 14.6, AB+ 0.99, O− 1.7, A− 0.5, B− 0.4, AB− 0.01.
  Fact 0.101: The World row has a population of 6,635,281,257, with blood type percentages O+ 38.67, A+ 27.42, B+ 22.02, AB+ 5.88, O− 2.55, A− 1.99, B− 1.11, AB− 0.36.

Retrieved items:
<item 1>
Blood type distribution by country
40 11 Spanish 38 47 10 5 Sudanese 62 16 21 0 Swedish 38 47 10 5 Swiss 40 50 7 Tatars 28 30 29 13 Thais 37 22 33 8 Turks 43 34 18 6 Ukrainians 37 40 18 6 Black
Americans 49 27 20 White Americans 45 40 11 Vietnamese 42 22 30 5 Mean 43.91 34.80 16.55 5.14 Standard deviation 16.87 13.80 9.97 3.41 Blood group B has its
highest frequency in South Asia where it ranks first as the largest share of the earth 's population . In Southeast Asia the share of the population is high ,
especially in Thailand and Indonesia , then in East Asia , Northern Asia and neighboring Central Asia , and its incidence diminishes both towards the east and
the west , falling to single digit percentages in Netherlands , Norway , Portugal and Switzerland . It is believed to have been entirely absent from Native
American and Australian Aboriginal populations prior to the arrival of Europeans in those areas .
</item 1>

<item 2>
Blood type distribution by country
Retrieved 2017 - 11 - 23 . Jump up ^ `` Veregruppidest '' ( Blood groups ) . verekeskus.ee ( in Estonian ) . Retrieved 2017 - 11 - 23 . Jump up ^ ``
Suomalaisten veriryhmäjakauma '' ( in Finnish ) . 2009 - 08 - 21 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Les groupes sanguins ( système ABO ) '' . Centre
Hospitalier Princesse GRACE - Monaco ( in French ) . C.H.P.G. MONACO. 2005 . Retrieved 2008 - 07 - 15 . Jump up ^ `` Blóðflokkar '' ( in Icelandic ) ...
landspitali.is . Archived from the original on 2011 - 07 - 19 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Blood Type Frequencies by Country including the Rh
Factor '' .
</item 2>

<item 3>
Blood type distribution by country
975 48.0 % 27.7 % 15.2 % 2.8 % 3.5 % 1.8 % 0.8 % 0.2 % Sweden 9,920,843 32.0 % 37.0 % 10.0 % 5.0 % 6.0 % 7.0 % 2.0 % 1.0 % Switzerland 8,454,321 35.0 % 40.0 %
7.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Syria 18,920,698 43.0 % 30.0 % 14.0 % 3.7 % 5.0 % 3.0 % 1.0 % 0.3 % Taiwan 23,234,936 43.9 % 25.9 % 23.9 % 6.0 % 0.17 % 0.1
% 0.01 % 0.02 % Thailand 68,298,027 40.8 % 16.9 % 36.
</item 3>

<item 4>
Blood type distribution by country
7 % 15.3 % 6.8 % 4.8 % 6.3 % 2.7 % 1.2 % Slovenia 2,071,258 31.0 % 33.0 % 12.0 % 6.0 % 7.0 % 7.0 % 3.0 % 1.0 % South Africa 55,437,815 39.0 % 32.0 % 12.0 % 3.0
% 6.0 % 5.0 % 2.0 % 1.0 % South Korea 50,748,307 27.9 % 33.87 % 26.92 % 10.98 % 0.1 % 0.13 % 0.08 % 0.02 % Spain 46,070,163 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0
% 2.0 % 0.5 % Sudan 40,827,975 48.0 % 27.7 % 15.2 % 2.
</item 4>

<item 5>
Blood type distribution by country
0 % 5.0 % 6.0 % 2.0 % 1.0 % Russia 143,374,801 28.0 % 30.0 % 20.0 % 7.0 % 4.9 % 5.8 % 3.2 % 1.1 % Saudi Arabia 32,744,532 47.8 % 23.9 % 17.0 % 4.0 % 4.0 % 2.0 %
1.0 % 0.3 % Serbia 8,776,827 31.92 % 35.28 % 12.6 % 4.2 % 6.08 % 6.72 % 2.4 % 0.8 % Singapore 5,784,819 43.6 % 23.9 % 24.4 % 6.0 % 0.9 % 0.7 % 0.4 % 0.1 %
Slovakia 5,432,265 27.2 % 35.7 % 15.3 % 6.8 % 4.8 % 6.
</item 5>

<item 6>
Blood type distribution by country
0 % 1.0 % Germany 80,635,980 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Greece 10,892,849 37.4 % 32.9 % 11.0 % 3.7 % 7.0 % 5.0 % 2.0 % 1.0 % Guinea
13,291,741 46.88 % 21.64 % 22.86 % 4.52 % 2.0 % 0.9 % 1.0 % 0.2 % Honduras 9,348,898 57.5 % 27.0 % 7.8 % 2.5 % 2.7 % 1.7 % 0.6 % 0.2 % Hong Kong 7,402,115 41.5
% 26.13 % 25.34 % 6.35 % 0.32 % 0.17 % 0.14 % 0.05 % Hungary 9,787,
</item 6>

<item 7>
Blood type distribution by country
17 % 0.14 % 0.05 % Hungary 9,787,801 27.0 % 33.0 % 16.0 % 8.0 % 5.0 % 7.0 % 3.0 % 1.0 % Iceland 334,311 47.6 % 26.4 % 9.3 % 1.6 % 8.4 % 4.6 % 1.7 % 0.4 % India
1,342,561,902 27.85 % 20.8 % 38.14 % 8.93 % 1.43 % 0.57 % 1.79 % 0.49 % Indonesia 263,519,317 36.82 % 25.87 % 28.85 % 7.96 % 0.18 % 0.13 % 0.15 % 0.04 % Iran
80,982,137 33.5 % 27.0 % 22.2 % 7.0 % 4.0 % 3.0 % 2.5 % 0.
</item 7>

<item 8>
Blood type distribution by country
Retrieved 2010 - 11 - 19 . Jump up ^ `` Federación Nacional de Donantes de Sangre / La sangre / Grupos '' . Donantesdesangre.net . Retrieved 2010 - 11 - 19 .
Jump up ^ `` Frequency of major blood groups in the Swedish population '' . Geblod.nu. 2007 - 10 - 02 . Archived from the original on 24 November 2010 .
Retrieved 2010 - 11 - 19 . Jump up ^ - BLUTSPENDE https://www.blutspende.ch/fr/don_de_sang/sanguin_groupe/groupe_sanguin - BLUTSPENDE Check url = value ( help )
. Missing or empty title = ( help ) Jump up ^ `` Turkey Blood Group Site '' . Kangrubu.com . Archived from the original on 29 May 2010 . Retrieved 2010 - 11 -
19 . Jump up ^ `` Blood Group Basics '' . Blood.co.uk . Retrieved 2017 - 07 - 30 .
</item 8>

<item 9>
Blood type distribution by country
2 % El Salvador 6,171,483 62.0 % 23.0 % 11.0 % 1.0 % 1.0 % 1.0 % 0.7 % 0.3 % Estonia 1,305,745 29.5 % 30.8 % 20.7 % 6.3 % 4.3 % 4.5 % 3.0 % 0.9 % Ethiopia
104,352,727 39.0 % 28.0 % 21.0 % 5.0 % 3.0 % 2.0 % 1.0 % 1.0 % Finland 5,541,328 27.0 % 38.0 % 15.0 % 7.0 % 4.0 % 6.0 % 2.0 % 1.0 % France 64,939,560 36.0 %
37.0 % 9.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Germany 80,635,980 35.
</item 9>

<item 10>
Blood type distribution by country
In countries such as Austria , Denmark , Norway , and Switzerland , about 45 -- 50 % of the population have this blood type , whereas about 40 % of Poles and
Ukrainians do so . The highest frequencies are found in small , unrelated populations . For example , about 80 % of the Blackfoot Indians of Montana have blood
type A ... Jump up ^ Technical Monograph No. 2 : The ABO Blood Group System and ABO Subgroups ( PDF ) . Biotec . March 2005 . Archived from the original ( PDF )
on 6 February 2007 ... The frequency of blood group A is quite high ( 25 -- 55 % ) in Europe , especially in Scandinavia and parts of central Europe . High
group A frequency is also found in the Aboriginal people of South Australia ( up to 45 % ) and in certain American Indian tribes where the frequency reaches 35
% ...
</item 10>
=== END ITEM id=b5b1157936726767 ===

=== ITEM id=ae618be2c9d22f3b ===
Gold sentences and facts:
Sentence 0: Loyalists were American colonists who remained loyal to the British Crown during the American Revolutionary War . At the time they were often called Tories , Royalists , or King 's Men . They were opposed by the Patriots , those who supported the revolution and called them `` persons inimical to the liberties of America '' . Prominent Loyalists repeatedly assured the British government that many thousands of loyalists would spring to arms and fight for the crown . The British government acted in expectation of that , especially in the southern campaigns in 1780 - 81 . In practice , the number of loyalists in military service was far lower than expected . Across the colonies , Patriots watched suspected Loyalists very closely , and would not tolerate any organized Loyalist opposition . Many outspoken or militarily active loyalists were forced to flee , especially to their stronghold of New York City . William Franklin , the royal governor of New Jersey and son of Patriot leader Benjamin Franklin , became the leader of the Loyalists after his release from a Patriot prison in 1778 . He worked to build Loyalist military units to fight in the war , but the number of volunteers was much fewer than London expected .
  Fact 0.0: Loyalists were American colonists who remained loyal to the British Crown during the American Revolutionary War.
  Fact 0.1: Loyalists were called Tories, Royalists, or King's Men at the time.
  Fact 0.2: Loyalists were opposed by the Patriots.
  Fact 0.3: The Patriots were those who supported the revolution.
  Fact 0.4: The Patriots called the Loyalists "persons inimical to the liberties of America."
  Fact 0.5: Prominent Loyalists repeatedly assured the British government that many thousands of loyalists would spring to arms and fight for the Crown.
  Fact 0.6: The British government acted in expectation of that, especially in the southern campaigns in 1780-81.
  Fact 0.7: In practice, the number of loyalists in military service was far lower than expected.
  Fact 0.8: Patriots watched suspected Loyalists very closely across the colonies.
  Fact 0.9: Patriots would not tolerate any organized Loyalist opposition.
  Fact 0.10: Many outspoken or militarily active loyalists were forced to flee.
  Fact 0.11: Loyalists fled especially to their stronghold of New York City.
  Fact 0.12: William Franklin was the royal governor of New Jersey.
  Fact 0.13: William Franklin was the son of Patriot leader Benjamin Franklin.
  Fact 0.14: William Franklin became the leader of the Loyalists after his release from a Patriot prison in 1778.
  Fact 0.15: William Franklin worked to build Loyalist military units to fight in the war.
  Fact 0.16: The number of volunteers for William Franklin's Loyalist military units was much fewer than London expected.

Retrieved items:
<item 1>
Loyalist (American Revolution)
They were alienated when the Patriots resorted to violence , such as burning houses and tarring and feathering . They wanted to take a middle - of - the road
position and were angry when forced by the Patriots to declare their opposition . They had a long - standing sentimental attachment to Britain ( often with
business and family links ) . They were procrastinators who realized that independence was bound to come someday , but wanted to postpone the moment . They were
cautious and afraid that chaos and mob rule would result . Some were pessimists who lacked the confidence in the future displayed by the Patriots . Others
recalled the dreadful experiences of many Jacobite rebels after the failure of the last Jacobite rebellion as recently as 1745 who often lost their lands when
the Hanoverian government won . Other motives of the Loyalists included : They felt a need for order and believed that Parliament was the legitimate authority .
In New York , powerful families had assembled colony - wide coalitions of supporters , Men long associated with the De Lancey faction went along when its
leadership decided to support the crown .
</item 1>

<item 2>
The excerpt profiles American Revolution Loyalists from several colonies, including judges, lawyers, merchants, clergy, officials, and soldiers. Notable figures
include Benedict Arnold, who switched sides; Black Loyalist Boston King, who settled in Nova Scotia and helped found Freetown; New York historian Thomas Jones,
who fled to Britain; and officials such as Daniel Leonard, later Bermuda’s chief justice. Others served in Loyalist militias, supported the Crown, or resettled
in Canada.
</item 2>

<item 3>
Loyalist (American Revolution)
This took a heavy toll , putting many of them out of action for some time . There was a slave by the name of Boston King who joined the Loyalists and wound up
catching smallpox . Boston King and other soldiers who were sick were relocated to a different part of the camp so that they did not contaminate the healthy
soldiers . The survivors joined other British units and continued to serve throughout the war . Black colonials were often the first to come forward to
volunteer and a total of 12,000 African Americans served with the British from 1775 to 1783 . This factor had the effect of forcing the rebels to also offer
freedom to those who would serve in the Continental Army ; however , such promises were often reneged upon by both sides . Americans who gained their freedom by
fighting for the British became known as Black Loyalists . The British honored the pledge of freedom in New York City through the efforts of General Guy
Carleton who recorded the names of African Americans who had supported the British in a document called the Book of Negroes which granted freedom to slaves who
had escaped and assisted the British .
</item 3>

<item 4>
Loyalist (American Revolution)
David Farnsworth ( d . 1778 ) , British agent hanged for his participation in a plot to undermine the American economy by distributing counterfeit currency
Elizabeth Graeme Fergusson , female Loyalist who wrote poems lamenting her husband 's desertion of her . These poems shared the grief that she and other women
left behind were feeling . William Franklin ( ca . 1730 -- 1813 ) , Governor of New Jersey , son of Benjamin Franklin Rebecca Franks ( 1760 -- 1823 ) ,
prominent member of Loyalist society in Philadelphia , Pennsylvania during the American Revolution Shaddrick Furman , a Virginia free black , was a guide and
information source for British troops . Upon his capture by Patriots , he was beaten and blinded . He received a pension of £ 18 per year for life from the
Loyalist Claims Commission .
</item 4>

<item 5>
Loyalist (American Revolution)
They felt themselves to be weak or threatened within American society and in need of an outside defender such as the British Crown and Parliament . They had
been promised freedom from slavery by the British . They felt that being a part of the British Empire was crucial in terms of commerce and their business
operations . Loyalism and military operations ( edit ) See also : Loyalists fighting in the American Revolution In the opening months of the Revolutionary War ,
the Patriots laid siege to Boston , where most of the British forces were stationed . Elsewhere there were few British troops and the Patriots seized control of
all levels of government , as well as supplies of arms and gunpowder . These actions were not without resistance . Especially in New York , New Jersey , and
parts of North and South Carolina , there was considerable ambivalence about the Patriot cause . Vocal Loyalists , often with the encouragement and assistance
of royal governors , recruited people to their side . In the South Carolina backcountry Loyalist recruitment oustripped that of Patriots .
</item 5>

<item 6>
Loyalists included David Farnsworth, a British agent hanged in 1778 for distributing counterfeit currency; poet Elizabeth Graeme Fergusson, who mourned her
husband’s desertion and expressed the grief of women left behind; William Franklin, New Jersey’s governor and Benjamin Franklin’s son; and Rebecca Franks, a
prominent Philadelphia Loyalist. Shaddrick Furman, a free Black Virginian who guided and informed British troops, was captured, beaten, and blinded by Patriots
</item 6>

<item 7>
Loyalist (American Revolution)
Most of the English - speaking settlers had arrived following the British conquest of Canada in 1759 -- 1760 , and were unlikely to support separation from
Britain . The older British colonies , Newfoundland and Nova Scotia ( including what is now New Brunswick ) also remained loyal and contributed military forces
in support of the Crown . In late 1775 the Continental Army sent a force into Quebec , led by General Richard Montgomery and Colonel Benedict Arnold , with the
goal of convincing the residents of Quebec to join the Revolution . Although only a minority of Canadians openly expressed loyalty to King George , about 1,500
militia fought for the King in the Siege of Fort St. Jean . In the region south of Montreal that was occupied by the Continentals , some inhabitants supported
the rebellion and raised two regiments to join the Patriot forces . In Nova Scotia , there were many Yankee settlers originally from New England , and they
generally supported the principles of the revolution . This element was declining in relative numbers and influence due to an influx of recent immigration from
the British isles , and they remained neutral during the war , and the influx was greatest in Halifax .
</item 7>

<item 8>
Loyalist (American Revolution)
1 A 9.2 B 9.3 C 9.4 D 9.5 E 9.6 F 9.7 G 9.8 H 9.9 I 9.10 J 9.11 K 9.12 L 9.13 M 9.14 N 9.15 O 9.16 P 9.17 R 9.18 S 9.19 T 9.20 V 9.21 W 9.22 X , Y , Z 10 See
also 11 References 12 Further reading 12.1 Compiled volumes of biographical sketches 12.2 Studies of individual Loyalists 12.3 Primary sources and guides to
manuscripts and the literature 13 External links 13.1 Videos Background ( edit ) The American Revolution was a civil war based on who would rule in the Thirteen
Colonies . Families were often divided as war forced colonists to choose sides in a conflict that remained for many years uncertain . Colonists , especially
recent arrivals , often felt themselves to be both American and British , subjects of the King , still owing a loyalty to the mother country .
</item 8>

<item 9>
Loyalist (American Revolution)
George Washington 's Mount Vernon . Jump up ^ `` The Loyalists '' . Jump up ^ Calhoun ( 1973 ) Jump up ^ Georgia Encyclopædia . Jump up ^ Robert M. Calhoon , in
' A companion to the American Revolution ' , ( 2000 ) ; p 235 . Jump up ^ John Adams has sometimes been cited as having claimed , in an 1813 letter , that one -
third of Americans supported the revolution and one - third were against . However , the passage in question actually refers to the French Revolution of 1789 .
Robert D. Marcus ( 1971 ) . The American Scene : Varieties of American History . p. 147 . See also `` Only 1 / 3 of Americans Supported the American Revolution
? '' , by William Marina . 6 - 28 - 2004 . Retrieved on July 14 , 2008 . Jump up ^ Ray Raphael ( 2012 ) . A People 's History of the American Revolution . The
New Press . p. 393 .
</item 9>

<item 10>
Loyalist (American Revolution)
Historian Robert Middlekauff summarized scholarly research on the nature of Loyalist support as follows : The largest number of loyalists were found in the
middle colonies : many tenant farmers of New York supported the king , for example , as did many of the Dutch in the colony and in New Jersey . The Germans in
Pennsylvania tried to stay out of the Revolution , just as many Quakers did , and when that failed , clung to the familiar connection rather than embrace the
new . Highland Scots in the Carolinas , a fair number of Anglican clergy and their parishioners in Connecticut and New York , a few Presbyterians in the
southern colonies , and a large number of the Iroquois Indians stayed loyal to the king . Johnson Hall , seat of Sir John Johnson in the Mohawk Valley New York
City and Long Island were the British military and political base of operations in North America from 1776 to 1783 and had a large concentration of Loyalists ,
many of whom were refugees from other states . According to Calhoon , Loyalists tended to be older and wealthier , but there were also many Loyalists of humble
means .
</item 10>
=== END ITEM id=ae618be2c9d22f3b ===

=== ITEM id=75c96df9d269a63c ===
Gold sentences and facts:
Sentence 0: Teri Garr as Mrs. Ramsay , Alec 's mother
  Fact 0.0: Teri Garr played the role of Mrs. Ramsay
  Fact 0.1: Mrs. Ramsay is Alec's mother

Retrieved items:
<item 1>
The Black Stallion (film)
American Film Institute . Retrieved 2016 - 08 - 20 . External links ( edit ) Official website The Black Stallion on IMDb The Black Stallion at AllMovie The
Black Stallion at Rotten Tomatoes The Black Stallion at Box Office Mojo The Black Stallion at the TCM Movie Database The Black Stallion by Walter Farley Books
The Black Stallion book series Films The Black Stallion ( 1979 ) The Black Stallion Returns ( 1983 ) The Young Black Stallion ( 2003 ) Television The Adventures
of the Black Stallion ( 1990 - 1993 ) Related Cass Ole El Mokhtar Docs Keepin Time Films directed by Carroll Ballard The Black Stallion ( 1979 ) Never Cry Wolf
( 1983 ) Nutcracker : The Motion Picture ( 1986 ) Wind ( 1992 ) Fly Away Home ( 1996 ) Duma ( 2005 ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
The Black Stallion (film)
1 John Buchanan as Jockey No. 2 Kristen Vigard as Becky Fausto Tozzi as Rescue Captain John Karlsen as Archeologist Leopoldo Trieste as Priest Frank Cousins as
African Chieftain Don Hudson as Zaurog Marne Maitland as Drake Captain Tom Dahlgren as Veterinarian Horses ( edit ) Cass Ole , a champion Arabian stallion , was
featured in most of the movie 's scenes , with Fae Jur , another black Arabian stallion , being his main double . Fae Jur 's main scene is the one where Alec is
trying to gain the trust of the Black on the beach . Two other stunt doubles were used for running , fighting and swimming scenes . El Mokhtar , an Egyptian
Arabian racehorse , was the producers ' first choice to portray the Black , but they were unable to secure his services for the film from his owners , who
declined any offers . He does appear in The Black Stallion Returns , alongside Cass Ole , by which time the studio bought out the syndicate of owners in order
to secure El Mokhtar 's services .
</item 2>

<item 3>
The Black Stallion (film)
He falls behind , but the Black wo n't stop . When Alec regains his balance , the Black is well on his way to catch up with his opponents . Alec now encourages
the Black to run as fast as he can , remembering the wild rides on the island , as they catch up . The Black eventually wins by two lengths . Cast ( edit )
Kelly Reno as Alec Ramsay Mickey Rooney as Henry Dailey Teri Garr as Mrs. Ramsay , Alec 's mother Hoyt Axton as Mr. Ramsay , Alec 's father Clarence Muse as
Snoe Michael Higgins as Jim Neville Ed McNamara as Jake Doghmi Larbi as Arab John Burton as Jockey No. 1 John Buchanan as Jockey No.
</item 3>

<item 4>
The Black Stallion (film)
The hungry stallion finds himself unable to resist , but visibly struggles with his distrust for humans . Eventually , the hunger wins and he takes Alec 's
offer ; their bond has been sealed and the two are now inseparable . Alec even manages to ride the unbroken horse , after many times falling off the horse . One
day , a fishing ship arrives , rescuing both Alec and the stallion . Back home in Flushing , NY , Alec is given a hero 's welcome . The Black has a temporary
home in Alec 's backyard , but a garbage man not knowing that there is a wild horse in the backyard is chased by the Black , who races off down the street after
being spooked by a passing car . Alec chases after him through every part of town , but loses track of him . The next day , Alec meets Snoe ( and Napoleon ) who
tell him where the Black is . Alec finds the stallion in the barn of Henry Dailey , a retired jockey and racehorse trainer , who apparently spent all night
catching the Black . Henry allows Alec to keep the Black in his barn .
</item 4>

<item 5>
The Black Stallion (film)
After being rescued , they are set on entering a race challenging two champion horses . The film is adapted by Melissa Mathison , Jeanne Rosenberg and William
D. Wittliff . It is directed by Carroll Ballard . The movie stars Kelly Reno , Mickey Rooney , Teri Garr , Hoyt Axton , and the Arabian horse Cass Ole . The
film features music by Carmine Coppola , the father of Hollywood producer Francis Ford Coppola , who was the executive producer of the film . Contents ( hide )
1 Plot 2 Cast 2.1 Horses 3 Awards and honors 3.1 Academy Awards 3.2 Golden Globe Awards 3.3 British Academy Awards 3.4 LA Film Critics Awards 3.5 Others 4
Sequel , prequel and television series 5 See also 6 References 7 External links Plot ( edit ) In 1946 Alec Ramsey is aboard the steamer Drake off the coast of
North Africa , where he sees a wild black stallion being forced into a makeshift stable compartment and heavily restrained by ropes leading to his halter .
</item 5>

<item 6>
The Black Stallion (film)
Alec 's father rescues Alec from his attacker but the life jacket is lost and father & son are separated . The black stallion breaks free from his compartment
and then jumps into the sea . Alec himself is swept overboard by a gigantic wave . Once in the water , he swims toward the stallion and managed to grab hold of
the ropes that were previously restraining the stallion . Alec wakes on the shore of a deserted island and starts to explore . He finds the stallion caught in
his restraints with the ropes stuck between the rocks . With his knife , Alec manages to free the stallion once again and the stallion runs away . For a time ,
the two keep their distance . Alec discovers means to survive by catching fish and seaweed . As Alec suddenly faces a cobra eye to eye , the Black comes to the
rescue and kills the snake , only to run off again . By now , Alec decides to try to get closer to the horse and offer him some seaweed . The hungry stallion
finds himself unable to resist , but visibly struggles with his distrust for humans .
</item 6>

<item 7>
The Black Stallion (film)
In 2002 , it was selected for preservation in the United States National Film Registry by the Library of Congress as being `` culturally , historically , or
aesthetically significant '' . Also , the film is recognized by American Film Institute in these lists : 2001 : AFI 's 100 Years ... 100 Thrills -- Nominated
2005 : AFI 's 100 Years of Film Scores -- Nominated 2006 : AFI 's 100 Years ... 100 Cheers -- # 64 Sequel , prequel and television series ( edit ) The film was
followed in 1983 by a sequel , The Black Stallion Returns , which also starred Reno . There was also a television series called The Adventures of the Black
Stallion which aired from 1990 to 1993 and starred Mickey Rooney and Richard Ian Cox . In 2003 , a 50 - minute prequel called The Young Black Stallion , was
shot and released for IMAX theaters . See also ( edit ) The Black Stallion books References ( edit ) Jump up ^ `` The Black Stallion , Box Office Information ''
. The Numbers .
</item 7>

<item 8>
The Black Stallion (film)
wikipedia.org/w/index.php?title=The_Black_Stallion_(film)&oldid=836268326 '' Categories : 1979 films English - language films American children 's films Films
about horses United States National Film Registry films United Artists films American Zoetrope films Directorial debut films Films shot in Sardinia Films
directed by Carroll Ballard Films that won the Best Sound Editing Academy Award Horse racing films Screenplays by Melissa Mathison American films Films set on
islands Films set in 1946 Hidden categories : Pages using div col with deprecated parameters Talk Contents About Wikipedia Català Dansk Deutsch Español فارسی
Français Bahasa Indonesia Italiano Bahasa Melayu Nederlands 日本 語 Polski Português Русский Svenska 6 more Edit links This page was last edited on 13 April 2018 ,
at 18 : 25 . About Wikipedia
</item 8>

<item 9>
The Black Stallion (film)
Henry allows Alec to keep the Black in his barn . When Alec wonders how fast the Black is , Alec and Henry decide to train the Black for the racetrack , while
Henry teaches Alec how to be a jockey . The Black surprises Henry with his speed . Henry immediately starts plotting to get the Black into a match race between
two of the country 's current champions , to be held at Santa Anita , California . To do that , he sets up a secret demonstration at night where the press can
witness his speed , keeping the identity of Alec and the Black secret . The news about the mystery horse is soon widespread and the Black is entered into the
race . The race is the most anticipated horse - racing event of the year . Before the two champions and the Black enter the starting gate , the Black gets into
a fight with one of his opponents , wounding his leg . Alec does not see the wound until he is in the gate . As he dismounts , the bell rings and the horses
take off . Alec desperately tries to stay on his horse and trying to stop him . He falls behind , but the Black wo n't stop .
</item 9>

<item 10>
The Black Stallion (film)
The Black stallion ( film ) - wikipedia The Black stallion ( film ) Jump to : navigation , search The Black Stallion 1979 poster Directed by Carroll Ballard
Produced by Fred Roos Francis Ford Coppola Tom Sternberg Screenplay by Melissa Mathison Jeanne Rosenberg William D. Wittliff Based on The Black Stallion by
Walter Farley Starring Kelly Reno Mickey Rooney Teri Garr Music by Carmine Coppola Cinematography Caleb Deschanel Edited by Robert Dalva Production company
American Zoetrope Distributed by United Artists Release date October 17 , 1979 ( 1979 - 10 - 17 ) Running time 118 minutes Country United States Language
English Budget $2.7 million Box office $37.8 million The Black Stallion is a 1979 American adventure film based on the 1941 classic children 's novel The Black
Stallion by Walter Farley . But the film starts in 1946 , five years after the book was published . It tells the story of Alec Ramsey , who is shipwrecked on a
deserted island with a wild Arabian stallion whom he befriends .
</item 10>
=== END ITEM id=75c96df9d269a63c ===

=== ITEM id=91f3a0c85e385f78 ===
Gold sentences and facts:
Sentence 0: The formation of ATP from ADP and P is energetically unfavorable and would normally proceed in the reverse direction . In order to drive this reaction forward , ATPase couples ATP synthesis during cellular respiration to an electrochemical gradient created by the difference in proton ( H ) concentration across the mitochondrial membrane in eukaryotes or the plasma membrane in bacteria . During photosynthesis in plants , ATP is synthesized by ATPase using a proton gradient created in the thylakoid lumen through the thylakoid membrane and into the chloroplast stroma .
  Fact 0.0: The formation of ATP from ADP and P is energetically unfavorable.
  Fact 0.1: The formation of ATP from ADP and P would normally proceed in the reverse direction.
  Fact 0.2: ATPase couples ATP synthesis during cellular respiration to an electrochemical gradient in order to drive the reaction forward.
  Fact 0.3: The electrochemical gradient is created by the difference in proton (H) concentration across the mitochondrial membrane in eukaryotes.
  Fact 0.4: The electrochemical gradient is created by the difference in proton (H) concentration across the plasma membrane in bacteria.
  Fact 0.5: During photosynthesis in plants, ATP is synthesized by ATPase using a proton gradient.
  Fact 0.6: During photosynthesis in plants, the proton gradient is created in the thylakoid lumen through the thylakoid membrane.
  Fact 0.7: During photosynthesis in plants, the proton gradient passes into the chloroplast stroma.
Sentence 1: Located within the thylakoid membrane and the inner mitochondrial membrane , ATP synthase consists of two regions F and F. F causes rotation of F and is made of c - ring and subunits a , b , d , F6. F is made of α , β , γ , δ ( \ displaystyle \ alpha , \ beta , \ gamma , \ delta ) subunits . F has a water soluble par that can hydrolyze ATP . F on the other hand has mainly hydrophobic regions . F F creates a pathway for protons movement across the membrane .
  Fact 1.0: ATP synthase is located within the thylakoid membrane and the inner mitochondrial membrane.
  Fact 1.1: ATP synthase consists of two regions, F and F.
  Fact 1.2: The F region causes rotation of the F region.
  Fact 1.3: The F region is made of a c-ring and subunits a, b, d, and F6.
  Fact 1.4: The F region is made of alpha, beta, gamma, and delta subunits.
  Fact 1.5: The F region has a water soluble part that can hydrolyze ATP.
  Fact 1.6: The F region has mainly hydrophobic regions.
  Fact 1.7: The F and F regions together create a pathway for protons to move across the membrane.
Sentence 2: The crystal structure of the F showed alternating alpha and beta subunits ( 3 of each ) , arranged like segments of an orange around a rotating asymmetrical gamma subunit . According to the current model of ATP synthesis ( known as the alternating catalytic model ) , the transmembrane potential created by ( H+ ) proton cations supplied by the electron transport chain , drives the ( H+ ) proton cations from the intermembrane space through the membrane via the F region of ATP synthase . A portion of the F ( the ring of c - subunits ) rotates as the protons pass through the membrane . The c - ring is tightly attached to the asymmetric central stalk ( consisting primarily of the gamma subunit ) , causing it to rotate within the alpha beta of F causing the 3 catalytic nucleotide binding sites to go through a series of conformational changes that leads to ATP synthesis . The major F subunits are prevented from rotating in sympathy with the central stalk rotor by a peripheral stalk that joins the alpha beta to the non-rotating portion of F. The structure of the intact ATP synthase is currently known at low - resolution from electron cryo - microscopy ( cryo - EM ) studies of the complex . The cryo - EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F. Under the right conditions , the enzyme reaction can also be carried out in reverse , with ATP hydrolysis driving proton pumping across the membrane .
  Fact 2.0: The crystal structure of the F region showed alternating alpha and beta subunits, three of each.
  Fact 2.1: The alpha and beta subunits are arranged like segments of an orange around a rotating asymmetrical gamma subunit.
  Fact 2.2: The current model of ATP synthesis is known as the alternating catalytic model.
  Fact 2.3: In the alternating catalytic model, the transmembrane potential created by proton cations supplied by the electron transport chain drives proton cations from the intermembrane space through the membrane via the F region of ATP synthase.
  Fact 2.4: A portion of the F region, the ring of c-subunits, rotates as protons pass through the membrane.
  Fact 2.5: The c-ring is tightly attached to the asymmetric central stalk, which consists primarily of the gamma subunit.
  Fact 2.6: The c-ring causes the central stalk to rotate within the alpha and beta subunits of F.
  Fact 2.7: The rotation causes the 3 catalytic nucleotide binding sites to go through a series of conformational changes that leads to ATP synthesis.
  Fact 2.8: The major F subunits are prevented from rotating in sympathy with the central stalk rotor by a peripheral stalk.
  Fact 2.9: The peripheral stalk joins the alpha and beta subunits to the non-rotating portion of F.
  Fact 2.10: The structure of the intact ATP synthase is currently known at low resolution from electron cryo-microscopy (cryo-EM) studies of the complex.
  Fact 2.11: The cryo-EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F.
  Fact 2.12: Under the right conditions, the enzyme reaction can also be carried out in reverse.
  Fact 2.13: In reverse, ATP hydrolysis drives proton pumping across the membrane.

Retrieved items:
<item 1>
The excerpt lists topics related to ATP synthase, including its rotary mechanism, electron-transfer chains, oxidative phosphorylation, proton pumps, and roles
in chloroplasts and mitochondria. References include studies on F₀F₁-ATP synthase structure and rotation (2011), a biochemical review (2015), and a 1966 paper
on oxidative-phosphorylation enzymes.
</item 1>

<item 2>
ATP synthase
Large - enough quantities of ATP cause it to create a transmembrane proton gradient , this is used by fermenting bacteria that do not have an electron transport
chain , but rather hydrolyze ATP to make a proton gradient , which they use to drive flagella and the transport of nutrients into the cell . In respiring
bacteria under physiological conditions , ATP synthase , in general , runs in the opposite direction , creating ATP while using the proton motive force created
by the electron transport chain as a source of energy . The overall process of creating energy in this fashion is termed oxidative phosphorylation . The same
process takes place in the mitochondria , where ATP synthase is located in the inner mitochondrial membrane and the F - part projects into mitochondrial matrix
. The consumption of ATP by ATP - synthase pumps proton cations into the matrix . Evolution ( edit ) The evolution of ATP synthase is thought to have been
modular whereby two functionally independent subunits became associated and gained new functionality . This association appears to have occurred early in
evolutionary history , because essentially the same structure and activity of ATP synthase enzymes are present in all kingdoms of life .
</item 2>

<item 3>
ATP synthase
ATP synthase - wikipedia ATP synthase Jump to : navigation , search ATP Synthase Molecular model of ATP synthase determined by X-ray crystallography . Stator is
not shown here . Identifiers EC number 3.6. 3.14 CAS number 9000 - 83 - 3 Databases IntEnz IntEnz view BRENDA BRENDA entry ExPASy NiceZyme view KEGG KEGG entry
MetaCyc metabolic pathway PRIAM profile PDB structures RCSB PDB PDBe PDBsum Gene Ontology AmiGO / QuickGO ( show ) Search PMC articles PubMed articles NCBI
proteins ATP synthase is an enzyme that creates the energy storage molecule adenosine triphosphate ( ATP ) . ATP is the most commonly used `` energy currency ''
of cells for most organisms . It is formed from adenosine diphosphate ( ADP ) and inorganic phosphate ( P ) .
</item 3>

<item 4>
ATP synthase is a rotary molecular machine in mitochondrial and thylakoid membranes. Its membrane-embedded F₀ region conducts protons, rotating a c-ring and
central γ-stalk; the soluble F₁ region contains alternating α and β subunits, with three catalytic β sites. Rotation drives β sites through loose, tight, and
open conformations, enabling ADP and phosphate to form and release ATP. A peripheral stalk prevents the catalytic head from rotating. The process can reverse,
</item 4>

<item 5>
ATP synthase is a conserved, reversible rotary enzyme: proton flow through membrane-embedded F₀ rotates a c-ring and central stalk, driving F₁’s three catalytic
β sites through conformations that synthesize ATP; ATP hydrolysis can instead generate a proton gradient. It functions in mitochondria, bacteria and
chloroplasts. Research covers its evolution, assembly, structure, catalytic residues, oligomycin sensitivity and rotary mechanism, including high-resolution
bovine structures. The material also
</item 5>

<item 6>
ATP synthase
The overall reaction catalyzed by ATP synthase is : ADP + P + H ⇌ ATP + H O + H The formation of ATP from ADP and P is energetically unfavorable and would
normally proceed in the reverse direction . In order to drive this reaction forward , ATPase couples ATP synthesis during cellular respiration to an
electrochemical gradient created by the difference in proton ( H ) concentration across the mitochondrial membrane in eukaryotes or the plasma membrane in
bacteria . During photosynthesis in plants , ATP is synthesized by ATPase using a proton gradient created in the thylakoid lumen through the thylakoid membrane
and into the chloroplast stroma . ATP synthase consists of two main subunits , F and F , which has a rotational motor mechanism allowing for ATP production .
Because of its rotating subunit ATP Synthase is a molecular machine .
</item 6>

<item 7>
ATP synthase catalyzes ADP + phosphate → ATP, an energetically unfavorable reaction driven by a proton gradient across membranes. In mitochondria and bacteria,
respiration generates the gradient; in chloroplasts, photosynthetic proteins drive proton flow from the thylakoid lumen into the stroma. The enzyme’s F₀ and F₁
subunits form a rotating molecular motor. Its structure and mechanism are broadly conserved: E. coli has a comparatively simple form, while
</item 7>

<item 8>
ATP synthase
According to the current model of ATP synthesis ( known as the alternating catalytic model ) , the transmembrane potential created by ( H+ ) proton cations
supplied by the electron transport chain , drives the ( H+ ) proton cations from the intermembrane space through the membrane via the F region of ATP synthase .
A portion of the F ( the ring of c - subunits ) rotates as the protons pass through the membrane . The c - ring is tightly attached to the asymmetric central
stalk ( consisting primarily of the gamma subunit ) , causing it to rotate within the alpha beta of F causing the 3 catalytic nucleotide binding sites to go
through a series of conformational changes that leads to ATP synthesis . The major F subunits are prevented from rotating in sympathy with the central stalk
rotor by a peripheral stalk that joins the alpha beta to the non-rotating portion of F. The structure of the intact ATP synthase is currently known at low -
resolution from electron cryo - microscopy ( cryo - EM ) studies of the complex .
</item 8>

<item 9>
ATP synthase is an enzyme that produces adenosine triphosphate (ATP) from adenosine diphosphate (ADP) and inorganic phosphate. ATP serves as the primary
energy-storage and transfer molecule—the “energy currency”—in most organisms.
</item 9>

<item 10>
ATP synthase is reversible: fermenting bacteria can hydrolyze ATP to create a proton gradient that powers flagella and nutrient transport, while respiring
bacteria use the electron-transport-chain proton motive force to make ATP through oxidative phosphorylation. In mitochondria, the enzyme sits in the inner
membrane, with its F₁ part facing the matrix. It likely evolved early through the association of two independent modules, and its structure and activity are
conserved across life.
</item 10>
=== END ITEM id=91f3a0c85e385f78 ===

=== ITEM id=5558555b947feb09 ===
Gold sentences and facts:
Sentence 0: The coastline of the island consists of a combination of long beaches , dunes , red sandstone cliffs , salt water marshes and numerous bays and harbours . The beaches , dunes and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration which oxidizes upon exposure to the air . The geological properties of a white silica sand found at Basin Head are unique in the province ; the sand grains cause a scrubbing noise as they rub against each other when walked on . Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours . The magnificent sand dunes at Greenwich are of particular significance . The shifting , parabolic dune system is home to a variety of birds and rare plants and is also a site of significant archeological interest .
  Fact 0.0: The coastline of the island consists of a combination of long beaches, dunes, red sandstone cliffs, salt water marshes, and numerous bays and harbours.
  Fact 0.1: The beaches, dunes, and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration.
  Fact 0.2: That iron concentration oxidizes upon exposure to air.
  Fact 0.3: The geological properties of a white silica sand found at Basin Head are unique in the province.
  Fact 0.4: The sand grains at Basin Head cause a scrubbing noise as they rub against each other when walked on.
  Fact 0.5: Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours.
  Fact 0.6: The magnificent sand dunes at Greenwich are of particular significance.
  Fact 0.7: The shifting, parabolic dune system at Greenwich is home to a variety of birds and rare plants.
  Fact 0.8: The shifting, parabolic dune system at Greenwich is also a site of significant archeological interest.

Retrieved items:
<item 1>
Geography of Prince Edward Island
Geography of Prince Edward Island - wikipedia Geography of Prince Edward Island Jump to : navigation , search This article contains wording that promotes the
subject in a subjective manner without imparting real information . Please remove or replace such wording and instead of making proclamations about a subject 's
importance , use facts and attribution to demonstrate that importance . ( August 2017 ) ( Learn how and when to remove this template message ) Prince Edward
Island The coast of Prince Edward Island around Cavendish Geography Location Gulf of Saint Lawrence Area 5,683.91 km ( 2,194.57 sq mi ) Highest elevation 152 m
( 499 ft ) Administration Canada Province Prince Edward Island Largest settlement Charlottetown ( pop . 32,174 ) Demographics Population 139,407 ( 2008 ) Pop .
</item 1>

<item 2>
Geography of Prince Edward Island
( See also a list of communities in Prince Edward Island . ) The island 's landscape is pastoral : rolling hills , pristine forests , white sand beaches , ocean
coves and the famous red soil have given Prince Edward Island a reputation as a province of outstanding natural beauty . A number of laws have been passed by
the provincial government to attempt to preserve the landscape through regulation , although the lack of consistent enforcement and absence of province - wide
zoning and land - use planning has resulted in some aesthetically displeasing development in recent years . The island 's lush landscape has had a strong
bearing not only on its economy but also its culture . Author Lucy Maud Montgomery drew inspiration from the land during the late Victorian Era for the setting
of her classic novel Anne of Green Gables . Today , many of the same qualities that Montgomery and others found in the Island are enjoyed by tourists who visit
during all seasons . They enjoy a variety of leisure activities , including beaches , various golf courses , eco-tourism adventures , and simply touring the
countryside and enjoying cultural events in local communities around the island .
</item 2>

<item 3>
On the Origin of Species
The coastal sea life of the Atlantic and Pacific sides of Central America had almost no species in common even though the Isthmus of Panama was only a few miles
wide . His explanation was a combination of migration and descent with modification . He went on to say : `` On this principle of inheritance with modification
, we can understand how it is that sections of genera , whole genera , and even families are confined to the same areas , as is so commonly and notoriously the
case . '' Darwin explained how a volcanic island formed a few hundred miles from a continent might be colonised by a few species from that continent . These
species would become modified over time , but would still be related to species found on the continent , and Darwin observed that this was a common pattern .
Darwin discussed ways that species could be dispersed across oceans to colonise islands , many of which he had investigated experimentally . Chapter XII
continues the discussion of biogeography . After a brief discussion of freshwater species , it returns to oceanic islands and their peculiarities ; for example
on some islands roles played by mammals on continents were played by other animals such as flightless birds or reptiles .
</item 3>

<item 4>
Dead Sea
Encyclopedia of world geography . Facts on File . p. 237 . ISBN 9780816072293 . Jump up ^ `` Dead Sea - Composition of Dead Sea Water '' . Archived from the
original on 2013 - 11 - 04 . ^ Jump up to : `` 5 alliances shortlisted to execute Red - Dead 's phase I '' . The Jordan Times . 27 November 2016 . Retrieved 3
December 2016 . Jump up ^ David Bridger ; Samuel Wolk ( September 1976 ) . The New Jewish Encyclopedia . Behrman House , Inc. p. 109 . ISBN 978 - 0 - 87441 -
120 - 1 . Retrieved July 25 , 2011 . It was named the `` Dead Sea '' because of the fact that no living thing can exist there , since the water is extremely
salty and bitter . Jump up ^ See bitumen and asphalt for more about asphaltite . Jump up ^ `` Springs and quicksand at the Dead Sea '' . Retrieved August 27 ,
2008 .
</item 4>

<item 5>
Geography of Prince Edward Island
32,174 ) Demographics Population 139,407 ( 2008 ) Pop . density 24.53 / km ( 63.53 / sq mi ) Ethnic groups Scottish ( 38.0 % ) , English ( 28.7 % ) , Irish (
27.9 % ) , French ( 21.3 % ) , German ( 4.0 % ) , and Dutch ( 3.1 % ) Prince Edward Island 's geography is mostly pastoral with red soil , white sand , and
scattered communities . Known as the `` Garden of the Gulf '' , the island is located in the Gulf of Saint Lawrence north of Nova Scotia and east of New
Brunswick , with which it forms the Northumberland Strait . PEI consists of the capital city Charlottetown , as well as urban towns Cornwall and Stratford and a
developing urban fringe . A smaller urban area surrounds Summerside Harbor , situated on the southern shore 40 kilometers west of Charlottetown Harbor , and
consists primarily of the city of Summerside . As with all natural harbors on the island , Charlottetown and Summerside harbors are created by rias .
</item 5>

<item 6>
Geography of Prince Edward Island
The geological properties of a white silica sand found at Basin Head are unique in the province ; the sand grains cause a scrubbing noise as they rub against
each other when walked on . Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours . The magnificent
sand dunes at Greenwich are of particular significance . The shifting , parabolic dune system is home to a variety of birds and rare plants and is also a site
of significant archeological interest .
</item 6>

<item 7>
Geography of Prince Edward Island
See also ( edit ) Geography of Canada portal List of parks in Prince Edward Island List of islands of Prince Edward Island References ( edit ) External links (
edit ) Geography of the provinces and territories of Canada National Geography of Canada Centre of Canada Provinces British Columbia Alberta Saskatchewan
Manitoba Ontario Quebec New Brunswick Nova Scotia Prince Edward Island Newfoundland and Labrador Territories Yukon Northwest Territories Nunavut See also
Regions of Canada Islands Rivers Lakes Mountains National Parks Volcanoes Portal WikiProject Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Geography_of_Prince_Edward_Island&oldid=835998214 '' Categories : Geography of Prince Edward Island Hidden categories
: Articles with peacock terms from August 2017 All articles with peacock terms Talk Contents About Wikipedia Français Edit links This page was last edited on 12
April 2018 , at 01 : 54 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you
agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 7>

<item 8>
Ring of Fire
Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser 's Triangle Canadian Shield Interior Alaska - Yukon lowland taiga
Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic Archipelago Labrador Peninsula Gaspé Peninsula Avalon Peninsula Bay de
Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula ( Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay Hudson Bay James
Bay Greenland United States Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic Commonwealth West Midwest Upper Midwest Mountain States
Intermountain West Basin and Range Province Oregon Trail Mormon Corridor Calumet Region Southwest Old Southwest Llano Estacado Central United States Tallgrass
prairie South South Central Deep South Upland South Four Corners East Coast West Coast Gulf Coast Third Coast Coastal states Eastern United States Appalachia
Trans - Mississippi Great North Woods Great Plains Interior Plains Great Lakes Great Basin Great Basin Desert Acadia Ozarks Ark - La - Tex Waxhaws Siouxland
Twin Tiers Driftless Area Palouse Piedmont Atlantic coastal plain Outer Lands Black Dirt Region Blackstone Valley Piney Woods Rocky Mountains Mojave Desert The
Dakotas The Carolinas Shawnee Hills San Fernando Valley Tornado Alley North Coast Lost Coast Emerald Triangle San Francisco Bay
</item 8>

<item 9>
Dead Sea
Northwest British Isles English Channel Channel Islands Cotentin Peninsula Normandy Brittany Gulf of Lion Iberia Al - Andalus Baetic System Pyrenees Alpide belt
South Italian Peninsula Insular Italy Tuscan Archipelago Aegadian Islands Iberia Al - Andalus Baetic System Gibraltar Arc Southeastern Mediterranean Crimea
Alpide belt Germanic Celtic Slavic countries Uralic European Plain Eurasian Steppe Pontic -- Caspian steppe Wild Fields Pannonian Basin Great Hungarian Plain
Little Hungarian Plain Eastern Slovak Lowland Regions of North America Northern Eastern Canada Western Canada Canadian Prairies Central Canada Northern Canada
Atlantic Canada The Maritimes French Canada English Canada Acadia Acadian Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser
's Triangle Canadian Shield Interior Alaska - Yukon lowland taiga Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic
Archipelago Labrador Peninsula Gaspé Peninsula Avalon Peninsula Bay de Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula (
Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay Hudson Bay James Bay Greenland Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic
Commonwealth West Midwest Upper Midwest Mountain States Intermountain West Basin and Range Province Oregon Trail
</item 9>

<item 10>
Geography of Prince Edward Island
The highest point of land is located at Springton in Queens County ( 152 metres or 499 feet above sea level ) . Contents ( hide ) 1 Communities 2 Coastline 3
See also 4 References 5 External links Communities ( edit ) The smaller rural communities as well as the towns and villages throughout the province proudly
retain a slower - paced , old world flavor , something that factors heavily into Prince Edward Island 's popularity as a destination for relaxation . Most rural
communities on Prince Edward Island are based on small - scale agriculture , given that the size of farm properties is quite small when compared with other
areas in Canada . There is an increasing amount of industrial farming as older farm properties are consolidated and modernized . Coastline ( edit ) The
coastline of the island consists of a combination of long beaches , dunes , red sandstone cliffs , salt water marshes and numerous bays and harbours . The
beaches , dunes and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration which oxidizes upon exposure to the air .
</item 10>
=== END ITEM id=5558555b947feb09 ===

=== ITEM id=db2c6c6784f6847b ===
Gold sentences and facts:
Sentence 0: 12 -- Bart Cummings ( 1965 , 1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 )
  Fact 0.0: Bart Cummings has 12 Melbourne Cup wins.
  Fact 0.1: Bart Cummings won the Melbourne Cup in 1965.
  Fact 0.2: Bart Cummings won the Melbourne Cup in 1966.
  Fact 0.3: Bart Cummings won the Melbourne Cup in 1967.
  Fact 0.4: Bart Cummings won the Melbourne Cup in 1974.
  Fact 0.5: Bart Cummings won the Melbourne Cup in 1975.
  Fact 0.6: Bart Cummings won the Melbourne Cup in 1977.
  Fact 0.7: Bart Cummings won the Melbourne Cup in 1979.
  Fact 0.8: Bart Cummings won the Melbourne Cup in 1990.
  Fact 0.9: Bart Cummings won the Melbourne Cup in 1991.
  Fact 0.10: Bart Cummings won the Melbourne Cup in 1996.
  Fact 0.11: Bart Cummings won the Melbourne Cup in 1999.
  Fact 0.12: Bart Cummings won the Melbourne Cup in 2008.
Sentence 1: 12 -- Bart Cummings ( 1965 , 1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 ) 5 -- Etienne de Mestre ( 1861 , 1862 , 1867 , 1877 , 1878 ) 5 -- Lee Freedman ( 1989 , 1992 , 1995 , 2004 , 2005 )
  Fact 1.0: Bart Cummings has 12 Melbourne Cup wins.
  Fact 1.1: Bart Cummings won the Melbourne Cup in 1965.
  Fact 1.2: Bart Cummings won the Melbourne Cup in 1966.
  Fact 1.3: Bart Cummings won the Melbourne Cup in 1967.
  Fact 1.4: Bart Cummings won the Melbourne Cup in 1974.
  Fact 1.5: Bart Cummings won the Melbourne Cup in 1975.
  Fact 1.6: Bart Cummings won the Melbourne Cup in 1977.
  Fact 1.7: Bart Cummings won the Melbourne Cup in 1979.
  Fact 1.8: Bart Cummings won the Melbourne Cup in 1990.
  Fact 1.9: Bart Cummings won the Melbourne Cup in 1991.
  Fact 1.10: Bart Cummings won the Melbourne Cup in 1996.
  Fact 1.11: Bart Cummings won the Melbourne Cup in 1999.
  Fact 1.12: Bart Cummings won the Melbourne Cup in 2008.
  Fact 1.13: Etienne de Mestre has 5 Melbourne Cup wins.
  Fact 1.14: Etienne de Mestre won the Melbourne Cup in 1861.
  Fact 1.15: Etienne de Mestre won the Melbourne Cup in 1862.
  Fact 1.16: Etienne de Mestre won the Melbourne Cup in 1867.
  Fact 1.17: Etienne de Mestre won the Melbourne Cup in 1877.
  Fact 1.18: Etienne de Mestre won the Melbourne Cup in 1878.
  Fact 1.19: Lee Freedman has 5 Melbourne Cup wins.
  Fact 1.20: Lee Freedman won the Melbourne Cup in 1989.
  Fact 1.21: Lee Freedman won the Melbourne Cup in 1992.
  Fact 1.22: Lee Freedman won the Melbourne Cup in 1995.
  Fact 1.23: Lee Freedman won the Melbourne Cup in 2004.
  Fact 1.24: Lee Freedman won the Melbourne Cup in 2005.
Sentence 2: 1877 is also the year that the trainer Etienne de Mestre won his fourth Melbourne Cup with Chester owned by Hon . James White . In 1878 , as in previous years De Mestre fielded more than one horse . He entered the favourite Firebell ( owned by W.S. Cox ) who finished last , Chester ( owned by Hon . James White ) the previous year 's winner who fell , and Calamia ( owned by de Mestre ) who , though less fancied , won easily by two lengths . First prize was £ 1,790 , the crowd was 80,000 and there were 30 starters . De Mestre 's 1878 win with Calamia brought to 5 the number of Melbourne Cups he had won . This record was not to be matched for nearly 100 years when the trainer Bart Cummings won his fifth Melbourne Cup in 1975 . Bart Cummings , regarded as the best Australian horse trainer of all time , went on to win 12 Melbourne Cups to 2008 .
  Fact 2.0: In 1877, the trainer Etienne de Mestre won his fourth Melbourne Cup.
  Fact 2.1: In 1877, Etienne de Mestre won with Chester.
  Fact 2.2: Chester was owned by Hon. James White.
  Fact 2.3: In 1878, as in previous years, Etienne de Mestre fielded more than one horse.
  Fact 2.4: Etienne de Mestre entered the favourite Firebell in 1878.
  Fact 2.5: Firebell was owned by W.S. Cox.
  Fact 2.6: Firebell finished last in 1878.
  Fact 2.7: Etienne de Mestre entered Chester in 1878.
  Fact 2.8: Chester was owned by Hon. James White.
  Fact 2.9: Chester was the previous year's winner.
  Fact 2.10: Chester fell in 1878.
  Fact 2.11: Etienne de Mestre entered Calamia in 1878.
  Fact 2.12: Calamia was owned by Etienne de Mestre.
  Fact 2.13: Calamia was less fancied.
  Fact 2.14: Calamia won easily by two lengths in 1878.
  Fact 2.15: First prize in 1878 was £1,790.
  Fact 2.16: The crowd in 1878 was 80,000.
  Fact 2.17: There were 30 starters in 1878.
  Fact 2.18: De Mestre's 1878 win with Calamia brought to 5 the number of Melbourne Cups he had won.
  Fact 2.19: De Mestre's record of 5 Melbourne Cups was not matched for nearly 100 years.
  Fact 2.20: Bart Cummings won his fifth Melbourne Cup in 1975.
  Fact 2.21: Bart Cummings is regarded as the best Australian horse trainer of all time.
  Fact 2.22: Bart Cummings went on to win 12 Melbourne Cups, up to 2008.

Retrieved items:
<item 1>
Melbourne Cup
2003 -- The first Melbourne Cup Tour was conducted around Australia , and the biggest crowd , of 122,736 , is recorded at Flemington . 2005 -- Makybe Diva
became the only horse so far to win the Melbourne Cup three times . 2008 -- The `` Cup King '' , Bart Cummings , took his 12th win in the Melbourne Cup with
Viewed 2010 - 150th anniversary . Americain becomes the first French - trained horse to win the race , and Gerald Mosse the first French jockey . 2011 - French
based horse Dunaden wins from Red Cadeaux in the closest finish ever . 2012 - The first seven finishers in the race were bred in Ireland . 2013 - Trainer Gai
Waterhouse became the first Australian female trainer to train the winner of the race when Fiorente won . Her father , legendary trainer Tommy J. Smith was also
a Melbourne Cup Winning trainer ( Toparoa in 1955 and Just A Dash in 1981 ) . 2014 - Two horses die due to racing in the 2014 Melbourne Cup .
</item 1>

<item 2>
Melbourne Cup
Maree Lyndon became the first female to ride in the Melbourne Cup , when she partnered Argonaut Style in 1987 , in which she ran second last in the 21 horse
field . In 2004 , Makybe Diva became the first mare to win two cups , and also the first horse to win with different trainers , after David Hall moved to Hong
Kong and transferred her to the Lee Freedman stables . The 2005 Melbourne Cup was held before a crowd of 106,479 . Makybe Diva made history by becoming the only
horse to win the race three times . Trainer Lee Freedman said after the race , `` Go and find the youngest child on the course , because that 's the only person
here who will have a chance of seeing this happen again in their lifetime . '' Due to the 2007 Australian Equine influenza outbreak , believed to have been
started by a horse brought into Australia from Japan , neither Delta Blues nor Pop Rock participated in the 2007 Melbourne Cup . Both horses had been stabled in
Japan .
</item 2>

<item 3>
Melbourne Cup
1862 - Archer became the first horse to win the race twice . 1863 - The smallest ever Melbourne Cup field of only 7 runners contested the race after several
horses were scratched in sympathy with Etienne de Mestre 's champion Archer being ruled out of the race on a technicality . 1869 - The Victorian Racing Club
introduced the four - day Spring Racing Carnival format . 1870 - The race was postponed a week . 1875 -- The first time the race was held on the first Tuesday
in November . 1876 - The youngest jockey in the history of the race Peter St. Albans won on Briseis aged 13 ( officially ) , but actually 12 years 11 months 23
days . 1882 -- The first bookmakers were licensed at Flemington . 1888 -- The first Gold whip was presented to the winning Cup jockey ( Mick O'Brien ) . 1890 -
The largest ever Melbourne Cup field of 39 runners contested the race with the champion Carbine ( horse ) winning and setting a weight carrying record of 10 st
5 lb ( 66 kg ) , which is unlikely to ever be beaten .
</item 3>

<item 4>
Melbourne Cup
Subsequent foreign bred horses to win Cup were Backwood 1924 ; Phar Lap 1930 ; Wotan 1936 Belldale Ball 1980 ; At Talaq 1986 ; Kingston Rule 1990 ; Vintage Crop
1993 ; Jeune 1994 ; Media Puzzle 2002 ; Makybe Diva 2003 , 2004 , 2005 ; Americain 2010 and Dunaden 2011 . The 1938 Melbourne Cup was won by trainer Mrs. Allan
McDonald , who conditioned Catalogue . Mrs McDonald was a successful trainer in New Zealand , however , at the time women were not allowed to compete as
trainers in Australia so her husband 's name was officially recorded as the winning trainer . The 2001 edition was won by New Zealand mare Ethereal , trained by
Sheila Laxon , the first woman to formally train a Melbourne Cup winner . She also won the Caulfield Cup , a 2,400 metre race also held in Melbourne , and
therefore has won the `` Cups Double '' .
</item 4>

<item 5>
Melbourne Cup
Volcanic Productions . p. 389 . ISBN 0 - 646 - 44865 - X . Jump up ^ Cavanough , Maurice , `` The Melbourne Cup '' , Jack Pollard P / L , North Sydney , 1976
Jump up ^ `` The Most Famous Racehorses in History '' . Retrieved 4 February 2011 . ^ Jump up to : Flemington Archived 14 July 2011 at the Wayback Machine .
Jump up ^ The Masters Touch , Racing with Etienne de Mestre , Winner of 5 Melbourne Cups by Keith W. Paterson , Published by Keith W. Paterson ( Nowra ) ( 2008
) ( ISBN 9780646500287 ) , Page 46 Jump up ^ Race May 13th 2008 09 : 34 ( 2008 - 05 - 13 ) . `` Aboriginal jockeys who won Melbourne Cup '' . Racingwrite.com .
Retrieved 2014 - 02 - 01 . Jump up ^ `` The wonder filly and a boy who carved their names into history '' . The Sydney Morning Herald . 1 November 2004 .
</item 5>

<item 6>
Melbourne Cup
Bart Cummings , regarded as the best Australian horse trainer of all time , went on to win 12 Melbourne Cups to 2008 . Martini - Henry , the 1883 Melbourne Cup
winner In 1883 , the hardy New Zealand bred , Martini - Henry won the VRC Derby , the Melbourne Cup and on the following Monday retained his undefeated record
by winning Mares ' Produce Stakes . Phar Lap winning the Melbourne Cup Race from Second Wind and Shadow King on 5 November 1930 . Phar Lap , the most famous
horse in the world of his day , won the 1930 Melbourne Cup at 11 / 8 odds on , the shortest priced favourite in the history of the race . He had to be hidden
away at Geelong before the race after an attempt was made to shoot him and only emerged an hour before the race time of the Cup . Phar Lap also competed in 1929
and 1931 , but came 3rd and 8th respectively , despite heavy favouritism in both years . There are a few legends of the first Aboriginal jockey to ride a
Melbourne Cup .
</item 6>

<item 7>
Melbourne Cup
Jump up ^ `` ABC Bullion wins production rights for the prestigious Emirates Melbourne Cup Trophy '' . Retrieved 4 September 2017 . Jump up ^ Melbourne Cup Tour
Jump up ^ `` Melbourne Cup Tour : Melbourne Cup Carnival '' . Melbournecup.com. 2013 - 11 - 05 . Retrieved 2014 - 02 - 01 . ^ Jump up to : `` First Past the
Post : The Melbourne Cup of 1861 '' . Australian Dictionary of Biography . Retrieved 9 September 2013 . Jump up ^ `` How Some of Australia 's Top Races Got
Their Names '' . Logan Livestock Insurance Agency . Retrieved 9 September 2013 . ^ Jump up to : The Masters Touch , Racing with Etienne de Mestre , Winner of 5
Melbourne Cups by Keith W. Paterson , Published by Keith W. Paterson ( Nowra ) ( 2008 ) ( ISBN 9780646500287 ) Jump up ^ `` The history of the Cup dates back to
1861 , the resultant brain child of Captain Frederick Standish , one time VRC Chairman and former Chief Commissioner of Police in Victoria .
</item 7>

<item 8>
Melbourne Cup
The race was won by the George Hanlon trained 40 - 1 outsider Piping Lane ridden by John Letts and carrying only 48 kg . 1985 -- The first sponsored Melbourne
Cup , and the first million dollar Cup , with $650,000 for the winner . 1987 -- First female jockey to ride in the cup was Maree Lyndon on Argonaut Style .
Dunaden : 2011 Melbourne Cup winner , painted by Charles Church 1993 -- The Dermot Weld trained Irish gelding Vintage Crop became the first Northern hemisphere
trained horse to win the Melbourne Cup and the first Irish horse . 2001 -- Sheila Laxon , was the first woman trainer to officially win the Melbourne Cup .
However , Mrs. A. McDonald ( 1938 ) with Catalogue was really the first woman trainer to win . Women then could not be registered as trainers in Australia , and
it was her husband who was the registered trainer . Mrs. McDonald 's win was as a female trainer of a female owned horse . 2003 -- First Australian female
jockey to ride in the cup was Clare Lindop on Debben .
</item 8>

<item 9>
Melbourne Cup
Perhaps they should have kept their money , however . The outsider Briseis comfortably won by 1 length in the biggest field of all time , and in an extremely
good time , so it is unlikely that the horses who perished could have beaten her . 1877 is also the year that the trainer Etienne de Mestre won his fourth
Melbourne Cup with Chester owned by Hon . James White . In 1878 , as in previous years De Mestre fielded more than one horse . He entered the favourite Firebell
( owned by W.S. Cox ) who finished last , Chester ( owned by Hon . James White ) the previous year 's winner who fell , and Calamia ( owned by de Mestre ) who ,
though less fancied , won easily by two lengths . First prize was £ 1,790 , the crowd was 80,000 and there were 30 starters . De Mestre 's 1878 win with Calamia
brought to 5 the number of Melbourne Cups he had won . This record was not to be matched for nearly 100 years when the trainer Bart Cummings won his fifth
Melbourne Cup in 1975 .
</item 9>

<item 10>
Melbourne Cup
1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 ) 5 -- Etienne de Mestre ( 1861 , 1862 , 1867 , 1877 , 1878 ) 5 -- Lee Freedman (
1989 , 1992 , 1995 , 2004 , 2005 ) Most wins by an owner ( edit ) 6 -- Lloyd Williams ( 1981 , 1985 , 2007 , 2012 , 2016 , 2017 ) 4 -- John Tait ( 1866 , 1868 ,
1871 , 1872 ) 4 -- Etienne de Mestre ( 1861 * , 1862 * , 1867 * , 1878 ) * The winners of these years were leased . 4 -- Dato Tan Chin Nam ( 1974 , 1975 , 1996
, 2008 ) Other records ( edit ) Record winning time -- 3.16.
</item 10>
=== END ITEM id=db2c6c6784f6847b ===
